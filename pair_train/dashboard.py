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
        htmlTable = html.HtmlTable()
        indexToNouncode = reverseMap(torch.load(open(nouncodeToIndexFile)))
        shardedDataHandler = gan.ShardedDataHandler(datasetDirectory)
        dashParzenHandler = gan.ShardedDataHandler(parzenDir, ".parzen")
        plotHandler = gan.ShardedDataHandler(trainJpgDir, ".log.jpg")
        ablationHandler = gan.ShardedDataHandler(ablationDir, ".ablation")
        htmlTable.addRow(
                "N1, N2", "Training Graph", "Parzen Fits", "Num Data Points",
                "Losses")
        for i, (n1, n2) in tqdm.tqdm(enumerate(shardedDataHandler.iterNounPairs())):
            tableRow = []
            intcodes = map(lambda x: indexToNouncode[x], (n1, n2))
            tableRow.append("%s\n%s\n%s" % (str((n1, n2)), str(intcodes), str(sdu.decodeNouns(*intcodes))))
            tableRow.append(html.ImgRef(
                    src='/%s' % os.path.relpath(
                            plotHandler.keyToPath((n1, n2)), ".")))
            tableRow.append(html.PhpTextFile(
                os.path.abspath(dashParzenHandler.keyToPath((n1, n2)))))
            tableRow.append("|data|=%d" % getNSamplesFromDatafile(shardedDataHandler.keyToPath((n1, n2))))

            tableRow.append(html.PhpTextFile('%s' % 
                os.path.abspath(ablationHandler.keyToPath((n1, n2)))))
            htmlTable.addRow(*tableRow)
        htmlMaker = html.HtmlMaker()

        htmlMaker.addElement(html.PhpTextFile(os.path.abspath(
                os.path.join(trainJpgDir, "args.txt"))))

        htmlMaker.addElement(htmlTable)
        htmlMaker.save(htmlOut)
