from multiprocessing import Pool, Queue
import sys
from methods import setOutputToFiles
import mylogger as logging
import tqdm

def target(arg):
    setOutputToFiles(str(arg))
    sys.stdout.write("STDOUT: %s\n" % str(arg))
    sys.stderr.write("STDERR: %s\n" % str(arg))
    print "print func %s" % str(arg)
    logging.getLogger(__name__).info("Logging %s" % str(arg))
    for _ in tqdm.tqdm(range(10), file=sys.stderr):
        print "print inside tqdm for %s" % str(arg)
    raise Exception

# If you use this, you probably want to set maxtasksperchild on your Pool, to
# ensure the opened files get closed.

def main():
    pool = Pool(None, maxtasksperchild=1) # TODO try with/without
    pool.map(target, range(4))
    pool.close(); pool.join() # TODO unnecessary

if __name__ == '__main__':
    main()
