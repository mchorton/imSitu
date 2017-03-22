import SimpleHTTPServer
import SocketServer
import os
import sys

def serve(ip, port, directory):
  print "Serving directory %s" % directory
  os.chdir(directory)
  print "Serving ip=%s, port=%d" % (ip, port)
  
  Handler = SimpleHTTPServer.SimpleHTTPRequestHandler
  httpd = SocketServer.TCPServer((ip, port), Handler)
  httpd.serve_forever()

if __name__ == '__main__':
  if len(sys.argv) != 4:
    print "Usage: progname ip port directory"
    sys.exit(1)
  serve(sys.argv[1], int(sys.argv[2]), sys.argv[3])
  sys.exit(0)

#PORT = 8000
#ip = "128.208.3.246"
