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
            self, parzenPath, trainJpgPath,
            datasetDirectory, nouncodeToIndexFile, htmlOut):
        htmlMaker = html.HtmlTable()
        indexToNouncode = reverseMap(torch.load(open(nouncodeToIndexFile)))
        shardedDataHandler = gan.ShardedDataHandler(datasetDirectory)
        dashParzenHandler = gan.ShardedDataHandler(parzenPath, ".parzen")
        plotHandler = gan.ShardedDataHandler(trainJpgPath, ".log.jpg")
        for i, (n1, n2) in tqdm.tqdm(enumerate(shardedDataHandler.iterNounPairs())):
            tableRow = []
            tableRow.append((n1, n2))
            intcodes = map(lambda x: indexToNouncode[x], (n1, n2))
            tableRow.append(intcodes)
            tableRow.append(sdu.decodeNouns(*intcodes))
            tableRow.append(html.ImgRef(src='"/%s"' % 
                os.path.relpath(plotHandler.keyToPath((n1, n2)), "data")))
            tableRow.append("TODO PARZEN") # This will need to be php... TODO
            tableRow.append("|data|=%d" % getNSamplesFromDatafile(shardedDataHandler.keyToPath((n1, n2))))

            tableRow.append("TODO ablation thing")
            htmlMaker.addRow(tableRow)
        logging.getLogger(__name__).info("output:")
        logging.getLogger(__name__).info(str(htmlMaker))
        with open(htmlOut, "w") as out:
            out.write(str(htmlMaker))
