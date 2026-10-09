"""Consistent, bounded SQLite snapshots, independent of the model/GPU."""
import argparse
from contextlib import closing
import fcntl
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

PREFIX='friday-memory-'

def backup(source,destination,keep=7,max_bytes=2*1024**3,min_free=2*1024**3):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if not source.is_file():raise ValueError('Memory database not found')
    if not 1<=keep<=30 or max_bytes<=0:raise ValueError('Invalid backup retention')
    destination.mkdir(parents=True,exist_ok=True,mode=0o700)
    os.chmod(destination,0o700)
    with (destination/'.backup.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Leave both the original and existing backups intact on insufficient space.
        estimate=source.stat().st_size+(source.with_name(source.name+'-wal').stat().st_size if source.with_name(source.name+'-wal').exists() else 0)
        if estimate>max_bytes:raise ValueError('Database exceeds the backup size budget')
        if shutil.disk_usage(destination).free < estimate+min_free:raise ValueError('Not enough drive space for a safe backup')
        fd,path=tempfile.mkstemp(prefix='.snapshot-',dir=destination);os.close(fd);temporary=Path(path)
        try:
            with closing(sqlite3.connect(source.as_uri()+'?mode=ro',uri=True)) as src,closing(sqlite3.connect(temporary)) as dst:
                src.backup(dst,pages=256,sleep=.05)
                dst.execute('PRAGMA journal_mode=DELETE')
                if dst.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('Snapshot integrity check failed')
            if temporary.stat().st_size>max_bytes:raise ValueError('Snapshot exceeds the backup size budget')
            os.chmod(temporary,0o600)
            with temporary.open('rb') as f:os.fsync(f.fileno())
            target=destination/(PREFIX+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+str(time.time_ns())+'.sqlite')
            temporary.replace(target)
            entries=sorted(destination.glob(PREFIX+'*.sqlite'),key=lambda p:p.name,reverse=True)
            total=0
            for index,path in enumerate(entries):
                total+=path.stat().st_size
                if index>=keep or total>max_bytes:path.unlink()
            dir_fd=os.open(destination,os.O_DIRECTORY)
            try:os.fsync(dir_fd)
            finally:os.close(dir_fd)
            return target
        finally:temporary.unlink(missing_ok=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--keep',type=int,default=7);parser.add_argument('--max-bytes',type=int,default=2*1024**3)
    parser.add_argument('--min-free',type=int,default=2*1024**3)
    args=parser.parse_args();target=backup(args.source,args.destination,args.keep,args.max_bytes,args.min_free)
    print('Memory snapshot created and integrity checked:',target.name)

if __name__=='__main__':main()
