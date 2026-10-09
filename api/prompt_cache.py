"""Keeps each chat request's start identical to the previous one, so llama.cpp can reuse its cache.

The phone app rebuilds every request from scratch: this turn's recalled memories are
appended to the system message, and the time note ("[Sent at 9:41 AM]") goes on the
newest user message only. Both change the start of the prompt every turn, and Qwen3.8's
hybrid (recurrent) layers then have to re-read the whole conversation, which took up to a
minute on the P100. Here the recalled memories move to the end of the newest user message,
and earlier user messages are sent exactly as they were the first time.

Chat from the app is pinned to one server slot and background work (memory extraction,
scheduled tasks, titles) to another, so background requests don't evict the chat's cache.
"""
import copy
import hashlib
import json
import re
from collections import OrderedDict

RECALL_MARKER = '\n\nRetrieved personal context.'
CHAT_SLOT = 0
BACKGROUND_SLOT = 1

_TIME_NOTE = re.compile(r'\s*\[Sent at [^\]]{1,20}\]\s*$')
_MAX_REMEMBERED = 2000
# Past this many characters, older turns are resent without their recalled memories, so a
# long chat doesn't fill the context; that costs one full re-read, then it's stable again.
LITE_AFTER_CHARS = 100_000
_sent: 'OrderedDict[str, tuple]' = OrderedDict()


def _without_note(content):
    if isinstance(content, str):
        return _TIME_NOTE.sub('', content)
    if isinstance(content, list):
        return [dict(p, text=_TIME_NOTE.sub('', p['text'])) if isinstance(p, dict) and isinstance(p.get('text'), str) else p
                for p in content]
    return content


def _with_recall(content, recall):
    if isinstance(content, str):
        return content + '\n\n' + recall
    if isinstance(content, list):
        return content + [{'type': 'text', 'text': recall}]
    return content


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(p.get('text', '') for p in content if isinstance(p, dict))
    return ''


def _new_recall(recall: str, already: str) -> str:
    """The recalled memories minus paragraphs the conversation already carries."""
    header, _, rest = recall.partition('\n\n')
    fresh = [p for p in rest.split('\n\n') if p.strip() and p.strip() not in already]
    return header + '\n\n' + '\n\n'.join(fresh) if fresh else ''


def stabilize(body: dict) -> dict:
    """Returns a copy of an OpenAI chat-completions body rearranged for cache reuse."""
    messages = body.get('messages')
    if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
        return body
    body = dict(body)
    messages = copy.deepcopy(messages)
    body['messages'] = messages
    body.setdefault('id_slot', CHAT_SLOT if body.get('tools') else BACKGROUND_SLOT)

    recall = ''
    first = messages[0]
    if first.get('role') == 'system' and isinstance(first.get('content'), str) and RECALL_MARKER in first['content']:
        stable, rest = first['content'].split(RECALL_MARKER, 1)
        first['content'] = stable
        recall = (RECALL_MARKER + rest).strip()

    users = [i for i, m in enumerate(messages) if m.get('role') == 'user']
    if not users:
        return body
    last = users[-1]
    digest = hashlib.sha256()
    earlier = {}
    for i, m in enumerate(messages):
        # Identify each user message by everything up to it, ignoring the parts that change per turn.
        normalized = dict(m, content=_without_note(m.get('content'))) if m.get('role') == 'user' else m
        digest.update(json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode())
        if m.get('role') != 'user':
            continue
        key = digest.hexdigest()
        if i < last:
            if key in _sent:
                earlier[i] = _sent[key]
                _sent.move_to_end(key)
        else:
            lite = copy.deepcopy(m.get('content'))
            if recall:
                m['content'] = _with_recall(m.get('content'), recall)
            _sent[key] = (copy.deepcopy(m['content']), lite)
            while len(_sent) > _MAX_REMEMBERED:
                _sent.popitem(last=False)
    full = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) + sum(
        len(json.dumps(f, ensure_ascii=False)) for f, _ in earlier.values())
    lite = full > LITE_AFTER_CHARS
    for i, (full_content, lite_content) in earlier.items():
        messages[i]['content'] = copy.deepcopy(lite_content if lite else full_content)
    if recall and not lite:
        # Memories recalled on an earlier turn are still in the conversation; don't send them twice.
        already = '\n'.join(_text(full_content) for full_content, _ in earlier.values())
        trimmed = _new_recall(recall, already)
        if trimmed != recall:
            content = _sent[key][1]
            messages[last]['content'] = _with_recall(content, trimmed) if trimmed else copy.deepcopy(content)
            _sent[key] = (copy.deepcopy(messages[last]['content']), _sent[key][1])
    return body


# ---- snapshots of the shared opening --------------------------------------
#
# Every new chat starts with the same system prompt and tool list (several thousand
# tokens). This model's cache can't keep just that part, so a new chat used to re-read
# it all (~45 s). A snapshot of the server's state after the opening is saved once
# (in the background slot, when chat is idle) and loaded into the chat slot whenever a
# new chat starts, which takes milliseconds.

import asyncio
import datetime
import logging
import os
import time
from pathlib import Path

log = logging.getLogger('uvicorn.error')
SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / 'slot-cache'
_LAST_SPEC = SNAPSHOT_DIR / 'last-opening.json'
_KEEP = 3
_MARK = '⁣next-message⁣'
_TEMPLATE_KEYS = ('tools', 'tool_choice', 'parallel_tool_calls', 'chat_template_kwargs')
_TODAY = re.compile(r'Today is (\w+), (\w+ \d{1,2}, \d{4})')


def _is_new_chat(messages) -> bool:
    roles = [m.get('role') for m in messages]
    return roles.count('user') == 1 and 'assistant' not in roles and 'tool' not in roles


def _opening(body: dict):
    messages = body.get('messages') or []
    if not messages or messages[0].get('role') != 'system' or not isinstance(messages[0].get('content'), str):
        return None
    spec = {k: body[k] for k in _TEMPLATE_KEYS if k in body}
    spec['messages'] = [messages[0]]
    return spec


def _file(spec) -> Path:
    key = hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
    return SNAPSHOT_DIR / f'opening-{key}.bin'


def _for_date(spec, day: datetime.date):
    text = spec['messages'][0]['content']
    if not _TODAY.search(text):
        return None
    dated = _TODAY.sub(f'Today is {day:%A}, {day:%B} {day.day}, {day.year}', text, count=1)
    return dict(spec, messages=[dict(spec['messages'][0], content=dated)])


class OpeningSnapshots:
    def __init__(self, client, gate):
        self.client = client
        self.gate = gate
        self.pending: list = []
        self.queued: set = set()
        # Openings someone is waiting for (a prime or a first message): built first, even during voice.
        self.urgent: set = set()
        self.done: dict = {}
        self.wake = asyncio.Event()
        self.tasks = []

    def start(self):
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        log.info('opening snapshot: background builder started')
        self.tasks = [asyncio.create_task(self._build_loop()), asyncio.create_task(self._daily_loop())]

    async def stop(self):
        for task in self.tasks:
            task.cancel()

    async def before_chat(self, body: dict):
        """Loads the opening's snapshot into the chat slot when a new chat starts."""
        if not body.get('tools') or body.get('id_slot') != CHAT_SLOT or not _is_new_chat(body.get('messages') or []):
            return
        spec = _opening(body)
        if spec is None:
            return
        self._remember(spec)
        path = _file(spec)
        if not path.exists():
            self._schedule(spec, urgent=True)
            # Its snapshot is coming: waiting for it beats re-reading the whole opening alongside it.
            try:
                await asyncio.wait_for(self.done[path.name].wait(), 120)
            except (asyncio.TimeoutError, KeyError):
                pass
            if not path.exists():
                return
        try:
            r = await self.client.post(f'/slots/{CHAT_SLOT}?action=restore', json={'filename': path.name}, timeout=30)
        except Exception as error:
            log.warning('opening snapshot: restore failed (%s); chat continues without it', type(error).__name__)
            return
        if r.status_code == 200:
            os.utime(path)
            return
        log.warning('opening snapshot: restore of %s refused (%s): %s', path.name, r.status_code, r.text[:200])
        if r.status_code == 400 and 'busy' not in r.text.lower():
            # A snapshot from another server build or settings: rebuild it.
            path.unlink(missing_ok=True)
            self._schedule(spec)

    def _remember(self, spec):
        try:
            tmp = _LAST_SPEC.with_suffix('.tmp')
            tmp.write_text(json.dumps(spec, ensure_ascii=False))
            tmp.replace(_LAST_SPEC)
        except OSError:
            pass

    def prime(self, body: dict) -> str:
        """An app's new-chat opening, sent before the first message: prepare its snapshot now."""
        if not body.get('tools'):
            return 'ignored'
        spec = _opening(body)
        if spec is None:
            return 'ignored'
        self._remember(spec)
        if _file(spec).exists():
            return 'ready'
        self._schedule(spec, urgent=True)
        return 'building'

    def _schedule(self, spec, urgent=False):
        name = _file(spec).name
        if urgent:
            self.urgent.add(name)
        if name in self.queued or _file(spec).exists():
            return
        self.queued.add(name)
        self.done[name] = asyncio.Event()
        if urgent:
            self.pending.insert(0, spec)
        else:
            self.pending.append(spec)
        self.wake.set()

    def _may_build(self, name) -> bool:
        gate = self.gate
        if name in self.urgent:
            # Someone is waiting: only an active reply or image render holds it back.
            return not gate.exclusive and gate.readers == 0 and not gate.maintenance_paused()
        return gate.background_allowed()

    async def _build_loop(self):
        while True:
            while not self.pending:
                self.wake.clear()
                await self.wake.wait()
            # Urgent openings first.
            self.pending.sort(key=lambda s: _file(s).name not in self.urgent)
            spec = self.pending.pop(0)
            name = _file(spec).name
            try:
                for attempt in range(3):
                    while not self._may_build(name):
                        await asyncio.sleep(1)
                    try:
                        started = time.monotonic()
                        await self._build(spec)
                        log.info('opening snapshot: built %s in %.0f s%s', name, time.monotonic() - started, ' (requested)' if name in self.urgent else '')
                        break
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        log.warning('opening snapshot: build of %s failed (attempt %d): %s', name, attempt + 1, repr(error)[:300])
                        await asyncio.sleep(30)
            finally:
                self.queued.discard(name)
                self.urgent.discard(name)
                event = self.done.pop(name, None)
                if event:
                    event.set()

    async def _build(self, spec):
        path = _file(spec)
        if path.exists():
            return
        r = await self.client.post('/apply-template', json=dict(spec, messages=spec['messages'] + [{'role': 'user', 'content': _MARK}]), timeout=30)
        r.raise_for_status()
        prompt = r.json()['prompt']
        cut = prompt.rfind('<|im_start|>user', 0, prompt.index(_MARK))
        if cut <= 0:
            return
        r = await self.client.post('/completion', json={'prompt': prompt[:cut], 'n_predict': 0, 'id_slot': BACKGROUND_SLOT, 'cache_prompt': True}, timeout=None)
        r.raise_for_status()
        r = await self.client.post(f'/slots/{BACKGROUND_SLOT}?action=save', json={'filename': path.name}, timeout=60)
        r.raise_for_status()
        snapshots = sorted(SNAPSHOT_DIR.glob('opening-*.bin'), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in snapshots[_KEEP:]:
            old.unlink(missing_ok=True)

    async def _daily_loop(self):
        """Prepares today's and tomorrow's openings from the last one seen, since the date is part of it."""
        while True:
            try:
                spec = json.loads(_LAST_SPEC.read_text())
                today = datetime.date.today()
                for day in (today, today + datetime.timedelta(days=1)):
                    dated = _for_date(spec, day)
                    if dated:
                        self._schedule(dated)
            except (OSError, ValueError, KeyError, IndexError):
                pass
            now = datetime.datetime.now()
            tomorrow = datetime.datetime.combine(now.date() + datetime.timedelta(days=1), datetime.time(0, 5))
            # Check again within the hour, so a missing snapshot doesn't stay missing all day.
            await asyncio.sleep(min((tomorrow - now).total_seconds(), 3600))
