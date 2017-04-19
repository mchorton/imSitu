import os
import split.data_utils as sdu
import split.v2pos.htmlgen as html
import torch
import tqdm
import re
import utils.mylogger as logging
import utils.methods as mt
import data

class GanDashboardMaker(object):
    def __init__(self):
        pass
    def makeDashboard(
            self, parzenDir, trainJpgDir, ablationDir,
            datasetDirectory, nouncodeToIndexFile, htmlOut):
        htmlTable = html.HtmlTable()
        indexToNouncode = mt.reverseMap(torch.load(open(nouncodeToIndexFile)))
        shardedDataHandler = data.ShardedDataHandler(datasetDirectory)
        dashParzenHandler = data.ShardedDataHandler(parzenDir, ".parzen")
        plotHandler = data.ShardedDataHandler(trainJpgDir, ".log.jpg")
        ablationHandler = data.ShardedDataHandler(ablationDir, ".ablation")
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
            tableRow.append("|data|=%d" % data.getNSamplesFromDatafile(shardedDataHandler.keyToPath((n1, n2))))

            tableRow.append(html.PhpTextFile('%s' % 
                os.path.abspath(ablationHandler.keyToPath((n1, n2)))))
            htmlTable.addRow(*tableRow)
        htmlMaker = html.HtmlMaker()

        htmlMaker.addElement(html.PhpTextFile(os.path.abspath(
                os.path.join(trainJpgDir, "args.txt"))))

        htmlMaker.addElement(htmlTable)
        htmlMaker.save(htmlOut)
