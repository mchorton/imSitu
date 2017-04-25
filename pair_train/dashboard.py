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
        ablationHandler = data.ShardedDataHandler(ablationDir, ".ablation")

        jpgConfig = [
            (data.ShardedDataHandler(trainJpgDir, ".predictions.jpg"), "Discriminator Performance"),
            (data.ShardedDataHandler(trainJpgDir, ".dgloss.jpg"), "Disc / Gen Loss"),
            (data.ShardedDataHandler(trainJpgDir, ".dgz.jpg"), "Change in Predictions on Fake Imgs"),
            (data.ShardedDataHandler(trainJpgDir, ".l1loss.jpg"), "L1 Loss"),
            (data.ShardedDataHandler(trainJpgDir, ".parzen.jpg"), "Parzen Fit"),
            (data.ShardedDataHandler(trainJpgDir, ".abl.jpg"), "Ablations")]

        plotHandler = data.ShardedDataHandler(trainJpgDir, ".log.jpg")
        parzenTrainimgHandler = data.ShardedDataHandler(trainJpgDir, ".parzen.jpg")
        ablTrainimgHandler = data.ShardedDataHandler(trainJpgDir, ".abl.jpg")

        labelrow = ["N1, N2", "N Data Points"] + list(zip(*jpgConfig)[1])
        htmlTable.addRow(*labelrow)

        for i, (n1, n2) in tqdm.tqdm(enumerate(shardedDataHandler.iterNounPairs())):
            tableRow = []
            intcodes = map(lambda x: indexToNouncode[x], (n1, n2))
            tableRow.append("%s\n%s\n%s" % (str((n1, n2)), str(intcodes), str(sdu.decodeNouns(*intcodes))))
            tableRow.append("|data|=%d" % data.getNSamplesFromDatafile(shardedDataHandler.keyToPath((n1, n2))))

            for maker, _ in jpgConfig:
                tableRow.append(html.ImgRef(
                        src='/%s' % os.path.relpath(
                                maker.keyToPath((n1, n2)), ".")))
            htmlTable.addRow(*tableRow)
        htmlMaker = html.HtmlMaker()

        htmlMaker.addElement(html.PhpTextFile(os.path.abspath(
                os.path.join(trainJpgDir, "args.txt"))))

        htmlMaker.addElement(htmlTable)
        htmlMaker.save(htmlOut)
