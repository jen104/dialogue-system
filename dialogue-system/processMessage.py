from dialogManager import dialogManager

## 通常会用redis缓存一个dm对像，这里直接用一个全局变量来处理
dm = dialogManager()

def processMessage(sentence):

    currentEvent = ''
    #先看是否有 还没有结束的场景  先处理最近的那个场景，因为在插入的时候，会往前插
    for event in dm.processing:
        # 获取实体 并填槽
        entities = dm.NerRecognizer(sentence).get("entities", {})
        if event == "checkWeather":
            if entities:
                city = ""
                time = ""
                for entity in entities:
                    type = entity.get("type", "")
                    if type in ["LOC"]:
                        city = entity.get("word", "")
                    if type in ["TIME"]:
                        time = entity.get("word", "")
                if city:
                    dm.slot["checkWeather"]["city"] = city
                if time:
                    dm.slot["checkWeather"]["time"] = time
                    currentEvent = "checkWeather"
            break
        elif event == "takeTaxi":
            if entities:
                place=""
                for entity in entities:
                    type = entity.get("type", "")
                    if type in ["LOC", "ORG"]:
                        place = entity.get("word", "")
                if place:
                    dm.slot["takeTaxi"]["destination"] = place
                currentEvent = "takeTaxi"
            break
        else:
            currentEvent = ""
            break

    ## 如果没有正在处理的场景
    if currentEvent == "":
        intent = dm.intentRecognizer(sentence)
        entities = dm.NerRecognizer(sentence).get("entities", {})
        if intent == "checkWeather":
            if entities:
                city=""
                time=""
                for entity in entities:
                    type = entity.get("type","")
                    if type in ["LOC"]:
                        city = entity.get("word", "")
                    if type in ["TIME"]:
                        time = entity.get("word", "")

                if city:
                    dm.slot["checkWeather"]["city"] = city
                if time:
                    dm.slot["checkWeather"]["time"] = time
            currentEvent = "checkWeather"
            dm.processing.insert(0,"checkWeather")
        elif intent == "takeTaxi":
            if entities:
                place=""
                for entity in entities:
                    type = entity.get("type", "")
                    if type in ["LOC", "ORG"]:
                        place = entity.get("word", "")
                    if not place:
                        place = entity.get("ORG", "")
                if place:
                    dm.slot["takeTaxi"]["destination"] = place
            currentEvent = "takeTaxi"
            dm.processing.insert(0, "takeTaxi")
        else:
            # 如果没有正在处理场景，但是意图识别不是在场景内，给一个兜底的回答给用户
            currentEvent = ""
            #print("主人，小安没有听懂您的意思，我会继续好好学习的。")
            return "主人，小安没有听懂您的意思，我会继续好好学习的。小安可以帮您查天气，打车。"

    ## 需要根据当前槽位的状况，决定采取什么样的动作和返回
    if currentEvent ==  "checkWeather":
        needTime = 0
        needCity = 0
        if not dm.slot["checkWeather"]["time"]:
            needTime = 1
        if not dm.slot["checkWeather"]["city"]:
            needCity = 1

        if needTime==1 and needCity==1:
            return "主人，请您告诉小安您要查询天气的城市和时间。"
        if needTime == 1 and needCity == 0:
            return "主人，请您告诉小安您要查询%s哪天的天气？"%city
        if needTime == 0 and needCity == 1:
            return "主人，请您告诉小安您要查询%s哪个城市的天气？"%time
        if needTime == 0 and needCity == 0:
            dm.processing.remove("checkWeather")
            return "主人，小安为您要查询到%s%s的天气为晴朗，微风阵阵"%(city,time)
    elif currentEvent ==  "takeTaxi":
        if not dm.slot["takeTaxi"]["destination"]:
            return "主人，请告诉小安您要去哪里？"
        else:
            dm.processing.remove("takeTaxi")
            return "主人，小安已经为您预定好去%s的车，请您做好准备" %(place)
    else:
        #print("主人，小安没有听懂您的意思，我会继续好好学习的。")
        return "主人，小安没有听懂您的意思，我会继续好好学习的。小安可以帮您查天气，打车。"



if __name__ == "__main__":
    print("主人，您好，小安可以帮您查天气，打车。")
    while True:
        sentence = input()
        print(processMessage(sentence))