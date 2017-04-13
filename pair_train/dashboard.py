import os
import split.data_utils as sdu
import split.v2pos.htmlgen as html
import torch
import gan
import tqdm
import re
import utils.mylogger as logging

def getNSamplesFromDatafile(filename):
    return len(torch.load(filename))

def reverseMap(mymap):
  return {v:k for k,v in mymap.iteritems()}

class GanDashboardMaker(object):
    def __init__(self):
        pass
    def makeDashboard(
            self, parzenDir, trainJpgDir, ablationDir,
            datasetDirectory, nouncodeToIndexFile, htmlOut):
        htmlMaker = html.HtmlTable()
        indexToNouncode = reverseMap(torch.load(open(nouncodeToIndexFile)))
        shardedDataHandler = gan.ShardedDataHandler(datasetDirectory)
        dashParzenHandler = gan.ShardedDataHandler(parzenDir, ".parzen")
        plotHandler = gan.ShardedDataHandler(trainJpgDir, ".log.jpg")
        ablationHandler = gan.ShardedDataHandler(ablationDir, ".ablation")
        for i, (n1, n2) in tqdm.tqdm(enumerate(shardedDataHandler.iterNounPairs())):
            tableRow = []
            tableRow.append((n1, n2))
            intcodes = map(lambda x: indexToNouncode[x], (n1, n2))
            tableRow.append(intcodes)
            tableRow.append(sdu.decodeNouns(*intcodes))
            tableRow.append(html.ImgRef(src='/%s' % 
                os.path.relpath(plotHandler.keyToPath((n1, n2)), "data")))
            tableRow.append(html.PhpTextFile('%s' % 
                os.path.abspath(dashParzenHandler.keyToPath((n1, n2)))))
            tableRow.append("|data|=%d" % getNSamplesFromDatafile(shardedDataHandler.keyToPath((n1, n2))))

            tableRow.append(html.PhpTextFile('%s' % 
                os.path.abspath(ablationHandler.keyToPath((n1, n2)))))
            htmlMaker.addRow(tableRow)
        with open(htmlOut, "w") as out:
            out.write(str(htmlMaker))
