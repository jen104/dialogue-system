
from intention.intent_recognition import *
from NER.NerPredict import evaluate_line

class dialogManager():

    def __init__(self):
        self.skillList=['checkWeather','takeTaxi']
        self.processing = [] # 代表正在处理的场景
        self.slot = {
            'checkWeather':{'time': "", "city":""},
            'takeTaxi':{"destination":"", "source":"您所在的地点"}
        }

    def intentRecognizer(self,sentence):
        predict_label = classifier.classify(get_word_features(sentence))
        return predict_label

    def NerRecognizer(self,sentence):
        nerResult = evaluate_line(sentence)
        return nerResult


if __name__ == "__main__":
    dm =dialogManager()
    print(dm.intentRecognizer("我要打车"))
    print(dm.NerRecognizer("福华路"))
