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

        stderrHandler = data.ShardedDataHandler(trainJpgDir, ".log.stderr")
        stdoutHandler = data.ShardedDataHandler(trainJpgDir, ".log.stdout")

        jpgConfig = [
            (data.ShardedDataHandler(trainJpgDir, ".stackloss.jpg"), "All Losses"),
            (data.ShardedDataHandler(trainJpgDir, ".valloss.jpg"), "Validation Loss"),
            (data.ShardedDataHandler(trainJpgDir, ".predictions.jpg"), "Discriminator Performance"),
            (data.ShardedDataHandler(trainJpgDir, ".valpredictions.jpg"), "Devset Discriminator Performance"),
            (data.ShardedDataHandler(trainJpgDir, ".dgz.jpg"), "Change in Predictions on Fake Imgs"),
            (data.ShardedDataHandler(trainJpgDir, ".l1loss.jpg"), "L1 Loss"),
            (data.ShardedDataHandler(trainJpgDir, ".devl1loss.jpg"), "Dev L1 Loss"),
            (data.ShardedDataHandler(trainJpgDir, ".parzen.jpg"), "Parzen Fit"),
            (data.ShardedDataHandler(trainJpgDir, ".devparzen.jpg"), "Dev Parzen Fit"),
            (data.ShardedDataHandler(trainJpgDir, ".rank.jpg"), "Mean Rank"),
            (data.ShardedDataHandler(trainJpgDir, ".abl.jpg"), "Ablations"),
            (data.ShardedDataHandler(trainJpgDir, ".devabl.jpg"), "Dev Ablations")]

        plotHandler = data.ShardedDataHandler(trainJpgDir, ".log.jpg")
        parzenTrainimgHandler = data.ShardedDataHandler(trainJpgDir, ".parzen.jpg")
        ablTrainimgHandler = data.ShardedDataHandler(trainJpgDir, ".abl.jpg")

        labelrow = ["N1, N2"] + list(zip(*jpgConfig)[1])
        htmlTable.addRow(*labelrow)

        for i, (n1, n2) in enumerate(shardedDataHandler.iterNounPairs()):
            tableRow = []
            intcodes = map(lambda x: indexToNouncode[x], (n1, n2))
            genInfo = "Noun IDS: %s\n" % str((n1, n2))
            genInfo += "Noun Codes: %s\n" % str(intcodes)
            genInfo += "Noun Translations: %s\n" % str(sdu.decodeNouns(*intcodes))
            genInfo += "|data|=%d\n" % data.getNSamplesFromDatafile(shardedDataHandler.keyToPath((n1, n2)))
            genInfo += "[%s] | [%s]" % (
                    str(html.HRef("/%s" % stderrHandler.keyToPath((n1, n2)), "stderr")),
                    str(html.HRef("/%s" % stdoutHandler.keyToPath((n1, n2)), "stdout")))

            tableRow.append(genInfo)
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
