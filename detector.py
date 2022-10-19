from calendar import c
from utils.cv.match import Vision
from utils.cv.reader import TextReader
import time
import numpy as np
import cv2 as cv
import threading
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath("./utils/cv")

    return os.path.join(base_path, relative_path)


def colorProcess(origin_img, lower, upper):  # filter color
    hsv_img = cv.cvtColor(origin_img, cv.COLOR_BGR2HSV)
    lower = np.array(lower)
    upper = np.array(upper)
    mask = cv.inRange(hsv_img, lower, upper)
    return mask


def getWhitePixels(origin_img, mask):  # get number of white pixels
    processed_img = cv.bitwise_and(origin_img, origin_img, mask=mask)
    processed_img = cv.cvtColor(processed_img, cv.COLOR_BGR2GRAY)
    white_pixel = cv.countNonZero(processed_img)
    return white_pixel


class Detector:
    def __init__(self, window_name=1):
        # get virtual camera from obs
        self.window_name = window_name

        # cap = cv.VideoCapture(self.window_name)
        # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1440)
        # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 810)

        # cv.imshow('check window', cap.read()[1])
        # cv.waitKey(0)

        self.__isRunning = False
        self.__screenshot = None
        self.__reader = None
        self.__ingame = False
        self.__closest_enemy_dis = 9999999
        self.__hp_value = 100
        self.__mp_value = 100
        self.__map_name = None
        self.__map_color = None
        self.__potion = 99999
        self.__coin = 999999
        self.__scroll_number = 100
        self.__spellscrolliszero = False
        self.__assist = False
        self.__recovery = False
        self.__attacked = False
        self.__strike = False
        self.__effect = False
        self.__crews = 0      # 隊友數量
        self.__crews_hp = []        # 隊友血量
        self.__crews_poisoned = []    # 隊友是否中毒
        self.__crews_freezed = []    # 隊友是否被石化
        self.__badge = Vision(resource_path("img\\badge.jpg"))
        self.__scroll = Vision(resource_path("img\\scroll.jpg"))
        self.__spell_scroll = Vision(resource_path("img\\spell_scroll.jpg"))
        self.__auto_move = Vision(resource_path("img\\auto_move.jpg"))
        self.__freezed = Vision(resource_path("img\\freezed.jpg"))

    def start(self):
        self.__reader = TextReader()

        print("使用虛擬攝影機編號: " + str(self.window_name))
        cap = cv.VideoCapture(self.window_name)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1440)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 810)
        ret, frame = cap.read()
        if ret:
            self.__screenshot = frame
            # self.__screenshot = cv.imread("C:\\Users\\YJack\\Documents\\VSCode_repo\\LineageW-CV\\v3\\img\\full2.jpg")
            cv.imshow("screen", self.__screenshot)
        else:
            print("cannot get screenshot")
            return False

        # 先設定 ingame 為 true 防止不測是名稱
        self.__ingame = True
        self.__updateMapName()
        print(self.getMapName())
        if self.getMapName() != None and self.getMapName() != "unknown":
            self.__ingame = True
            print("在遊戲中!")
        else:
            self.__ingame = False
            print("未在遊戲中!")
            return False

        self.__main_thread = threading.Thread(target=self.__fast_loop)
        self.__slow_thread = threading.Thread(target=self.__slow_loop)

        self.__isRunning = True
        self.__main_thread.start()
        self.__slow_thread.start()
        return True

    def stop(self):
        self.__isRunning = False
        self.__main_thread.join()
        self.__slow_thread.join()
        return

    # rapid detection on fast loop

    def __fast_loop(self):
        # get virtual camera from obs
        self.cap = cv.VideoCapture(self.window_name)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1440)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 810)

        while self.__isRunning:
            if not self.cap.isOpened():
                print("cap is not opened")
                continue
            ret, frame = self.cap.read()
            if ret:
                self.__screenshot = np.array(frame)
                # self.__screenshot = cv.imread("C:\\Users\\giorgio\\Documents\\GitHub\\LineageW-CV\\v3\img\\full2.jpg")
                self.__screenshot = self.__screenshot[0:810, 0:1440, :3]
                self.__screenshot = np.ascontiguousarray(self.__screenshot)
            else:
                # print("cap read error")
                continue

            cv.imshow('screen', self.__screenshot)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            # detect whether in game or not
            origin_ingame_img = self.__screenshot[765:785, 23:46]
            ingame_mask = colorProcess(origin_ingame_img, [0, 0, 167], [
                                       179, 116, 255])  # in game color filtering
            white_pixels = cv.countNonZero(ingame_mask)

            if white_pixels > 460 * 0.3:
                self.__ingame = True

                # enemy detect
                enemy = self.__badge.templateMatching(self.__screenshot, 0.8)
                # marked = self.__badge.drawCrosshairs(screenshot, enemy)
                x = np.array(enemy)
                x = [i-[720, 405] for i in x]
                minDis = 99999999
                for i in range(len(x)):
                    tmp = x[i][0] * x[i][0] + x[i][1] * x[i][1]
                    if tmp < minDis:
                        minDis = tmp

                self.__closest_enemy_dis = minDis

                # HP detect
                origin_hp_img = self.__screenshot[42:54, 123:390]
                hp_mask = colorProcess(
                    origin_hp_img, [0, 0, 0], [179, 142, 243])
                cv.imshow('hp', origin_hp_img)
                black_part = (getWhitePixels(
                    origin_hp_img, hp_mask)/3204) * 100
                self.__hp_value = 100 - black_part

                # MP detect
                origin_mp_img = self.__screenshot[62:72, 123:382]
                mp_mask = colorProcess(origin_mp_img, [19, 0, 31], [
                                       179, 255, 255])  # hp color filtering
                self.__mp_value = (getWhitePixels(
                    origin_mp_img, mp_mask)/2590) * 100
            else:
                self.__ingame = False

            self.__updateEffect()

        self.__screenshot = None
        print("getScreenshot finished")
        self.cap.release()
        return

    # get data from fast loop

    def getHP(self):
        return self.__hp_value

    def getMP(self):
        return self.__mp_value

    def getClosestEnemy(self):
        return self.__closest_enemy_dis

    def getMapColor(self):
        return self.__map_color

    # text detection from slow loop

    def __slow_loop(self):
        while self.__isRunning:
            self.__updateMapName()
            self.__updatePotion()
            self.__updateCoin()
            self.__updateScroll()
            self.__updateIsSpellscrollZero()
            self.__updateAssist()
            self.__updateRecovery()
            self.__updateAttacked()
            self.__updateStrike()  # 魔法卷軸按鈕是否觸發
            time.sleep(2)

    def getMapName(self):
        return self.__map_name

    def __updateMapName(self):
        if not self.__ingame:
            return

        map_icon = self.__screenshot[130:170, 1125:1165]
        self.__map_color, lower, upper = self.__getMapColorAndMask(map_icon)

        origin_map_img = self.__screenshot[130:180, 1130:1380]
        # resize to make text detecting easier
        larger_img = cv.resize(origin_map_img, (int(250*1.5), int(50*1.5)))
        map_mask = colorProcess(larger_img, lower, upper)
        processed_img = cv.bitwise_and(larger_img, larger_img, mask=map_mask)
        # convert to gray to make text detecting easier
        processed_img = cv.cvtColor(processed_img, cv.COLOR_BGR2GRAY)
        output = self.__reader.toText(processed_img)

        if len(output) > 0:
            self.__map_name = output[0][1].replace(" ", "")
        else:
            self.__map_name = "unknown"

        return

    def __getMapColorAndMask(self, map_icon):
        color = "unknown"
        lower, upper = np.array([0, 0, 0]), np.array([255, 255, 255])
        blue_mask_map_icon = colorProcess(
            map_icon, [70, 0, 0], [179, 255, 255])
        white_mask_map_icon = colorProcess(map_icon, [0, 0, 0], [179, 85, 77])

        if cv.countNonZero(blue_mask_map_icon) > 1600 * 0.1:  # mask blue out
            color = "blue"
            lower = np.array([61, 47, 121])
            upper = np.array([179, 255, 255])
        elif cv.countNonZero(white_mask_map_icon) > 1600 * 0.05:
            color = "white"
            lower = np.array([0, 0, 153])
            upper = np.array([179, 255, 255])
        else:
            color = "red"
            lower = np.array([0, 0, 130])
            upper = np.array([10, 255, 255])

        return color, lower, upper

    def matchMapName(self, target):
        # using LCS to get the similarity
        dp = np.zeros((len(self.__map_name)+1, len(target)+1), dtype=int)
        max_length = max(len(self.__map_name), len(target))
        for i in range(len(self.__map_name)+1):
            dp[i][0] = i
        for i in range(len(target)+1):
            dp[0][i] = i

        for i in range(1, len(self.__map_name)+1):
            for j in range(1, len(target)+1):
                x = dp[i-1][j] + 1
                y = dp[i][j-1] + 1
                if self.__map_name[i-1] == target[j-1]:
                    z = dp[i-1][j-1]
                else:
                    z = dp[i-1][j-1] + 1
                dp[i][j] = min(x, y, z)
        distance = dp[len(self.__map_name)][len(target)]
        accuracy = (max_length-distance) / max_length

        return accuracy >= 0.75

    def getPotion(self):
        return self.__potion

    def __updatePotion(self):
        if not self.__ingame:
            return

        origin_potion_img = self.__screenshot[730:755, 455:505]
        potion_mask = colorProcess(
            origin_potion_img, [0, 0, 190], [179, 255, 255])
        processed_img = cv.bitwise_and(
            origin_potion_img, origin_potion_img, mask=potion_mask)
        larger_potion_img = cv.resize(processed_img, (60, 30))
        output = self.__reader.toText(larger_potion_img)

        # when potion is 0, the red part of the img will be more, and white part will be less
        zero_potion_mask = colorProcess(
            origin_potion_img, [0, 0, 135], [179, 255, 255])
        white_pixels = cv.countNonZero(zero_potion_mask)

        if len(output) > 0:
            res = output[0][1] \
                .replace(" ", "") \
                .replace("[", "1") \
                .replace("@", "6") \
                .replace("|", "1") \
                .replace("{", "0") \
                .replace("}", "0") \
                .replace("`", "") \
                .replace(",", "")
            if res.isnumeric():
                self.__potion = int(res)
        elif white_pixels < 1250 * 0.07:
            self.__potion = 0
        else:
            print("Unable to detect potion")

        return

    def getScroll(self):
        return self.__scroll_number

    def __updateScroll(self):
        if not self.__ingame:
            return

        items_row = self.__screenshot[700:760, 1040:1390]
        scroll_pos = self.__scroll.templateMatching(items_row, 0.7)

        origin_scroll_img = None
        if len(scroll_pos) > 0:
            pos_x = scroll_pos[0][1]
            pos_y = scroll_pos[0][0]
            origin_scroll_img = self.__screenshot[700 +
                                                  pos_x: 740 + pos_x, 1040 + pos_y: 1080 + pos_y]
        else:
            print("no scroll detected")
            return

        scroll_mask = colorProcess(
            origin_scroll_img, [0, 0, 140], [179, 36, 255])
        processed_img = cv.bitwise_and(
            origin_scroll_img, origin_scroll_img, mask=scroll_mask)
        # resize img to make text detecting easier
        larger_scroll_img = cv.resize(processed_img, (80, 80))
        output = self.__reader.toText(larger_scroll_img)

        if len(output) > 0:
            res = output[0][1].replace(" ", "").replace(
                "[", "1").replace("|", "1")
            # print("scroll number: " + res)
            if res.isnumeric():
                self.__scroll_number = int(res)
        else:
            print("Unable to detect number of scrolls")

        return

    def getCoin(self):
        return self.__coin

    def __updateCoin(self):
        if not self.__ingame:
            return

        origin_coin_img = self.__screenshot[7:40, 610:745]
        coin_mask = colorProcess(origin_coin_img, [0, 0, 110], [
                                 179, 29, 255])  # coin color filtering
        processed_img = cv.bitwise_and(
            origin_coin_img, origin_coin_img, mask=coin_mask)
        larger_coin_img = cv.resize(processed_img, (176, 43))
        output = self.__reader.toText(larger_coin_img)
        if len(output):
            res = output[0][1].replace(" ", "").replace("[", "1").replace(
                "|", "1").replace(",", "").replace(".", "")
            if res.isnumeric():
                self.__coin = int(res)

        return

    def isSpellscrollZero(self):
        return self.__spellscrolliszero

    def __updateIsSpellscrollZero(self):  # 魔法卷軸是否為0
        if not self.__ingame:
            return

        items_row = self.__screenshot[700:760, 640:990]
        spell_scroll_num = self.__spell_scroll.templateMatching(items_row, 0.9)

        if len(spell_scroll_num) > 0:
            self.__spellscrolliszero = True
        else:
            self.__spellscrolliszero = False

        return

    def getAssist(self):
        return self.__assist

    def __updateAssist(self):
        if not self.__ingame:
            return

        origin_assist_img = self.__screenshot[427:496, 1296:1362]
        assist_mask = colorProcess(origin_assist_img, [0, 0, 255], [
                                   179, 255, 255])  # assist color filtering
        white_pixels = cv.countNonZero(assist_mask)

        if white_pixels >= 4554 * 0.01:
            self.__assist = True
        else:
            self.__assist = False

        return

    def getStrike(self):
        return self.__strike

    def __updateStrike(self):
        if not self.__ingame:
            return

        origin_strike_img = self.__screenshot[506:620, 1213:1323]
        strike_mask = colorProcess(origin_strike_img, [0, 78, 50], [
                                   34, 255, 255])  # strike color filtering
        white_pixels = cv.countNonZero(strike_mask)

        if white_pixels < 12540 * 0.1:
            self.__strike = False
        else:
            self.__strike = True

        return

    def getRecovery(self):
        return self.__recovery

    def __updateRecovery(self):
        if not self.__ingame:
            return

        origin_recovery_img = self.__screenshot[5:65, 925:990]
        recovery_mask = colorProcess(origin_recovery_img, [0, 90, 120], [
                                     179, 255, 255])  # recovery color filtering
        white_pixels = cv.countNonZero(recovery_mask)

        if white_pixels < 3900*0.1:
            self.__recovery = False
        else:
            self.__recovery = True

        return

    def getAttacked(self):
        return self.__attacked

    def __updateAttacked(self):
        if not self.__ingame:
            return

        origin_attacked_img = self.__screenshot[515:595, 1115:1205]
        attacked_mask = colorProcess(origin_attacked_img, [0, 61, 141], [
                                     179, 255, 255])  # attacked color filtering
        white_pixels = cv.countNonZero(attacked_mask)

        if white_pixels < 7200*0.05:
            self.__attacked = False
        else:
            self.__attacked = True

        return

    def getAutomove(self):
        automove = self.__auto_move.templateMatching(self.__screenshot, 0.9)
        if automove:
            automove_pos = automove_pos[0]
            return automove_pos
        else:
            return -1

    def getEffect(self):
        return self.__effect

    def __updateEffect(self):
        if not self.__ingame:
            return

        # mask off center part
        blank = np.full(self.__screenshot.shape[:2], 255).astype(np.uint8)
        rec_mask = cv.rectangle(blank, (732, 693), (798, 759), (0, 0, 0), -1)
        rec_masked = cv.bitwise_and(
            self.__screenshot, self.__screenshot, mask=rec_mask)
        origin_effect_img = rec_masked[690:762, 729:801]

        # filtering the white part, count the number of white pixels
        effect_mask = colorProcess(
            origin_effect_img, [0, 56, 101], [179, 151, 212])
        white_pixels = cv.countNonZero(effect_mask)

        if white_pixels:
            self.__effect = True
        else:
            self.__effect = False

        return
    
    def __updateCrews(self):  #更新隊友數量
        if not self.__ingame:
            return

        temp_crews = 0
        for i in range(3):
            origin_crews_img = self.__screenshot[205+(i*75):240+(i*75), 160:200]
            crews_mask = colorProcess(origin_crews_img, [0, 0, 240], [179, 21, 255])
            white_pixels = cv.countNonZero(crews_mask)
            if white_pixels > 1400:
                temp_crews += 1
        self.__crews = temp_crews
        
    def getCrews(self):
        return self.__crews
    
    def __updateCrewsHP(self):  #依照隊友數量更新血量
        if not self.__ingame:
            return
        self.__crews_hp = []
        for i in range(self.__crews):
            crew_hp_img = self.__screenshot[250+(i*75):256+(i*75), 130:290]
            crew_hp_mask = colorProcess(crew_hp_img, [0, 97, 100], [179, 255, 255])
            white_pixels = getWhitePixels(crew_hp_img, crew_hp_mask)
            self.__crews_hp.append(white_pixels/960)
        return 
    
    def getCrewsHP(self):
        return self.__crews_hp
    
    def __updateCrewsIsPoisoned(self):  #依照隊友數量更新中毒狀態
        if not self.__ingame:
            return
        self.__crews_poisoned = []
        for i in range(self.__crews):
            crew_poison_img = self.__screenshot[250+(i*75):256+(i*75), 130:290]
            crew_poison_mask = colorProcess(crew_poison_img, [21, 82, 73], [56, 255, 255])
            white_pixels = getWhitePixels(crew_poison_img, crew_poison_mask)
    
            if white_pixels > 960 * 0.02: # 2% hp of the bar is detected poisoned
                self.__crews_poisoned.append(1)
            else:
                self.__crews_poisoned.append(0)
        return 
            
    def getCrewsIsPoisoned(self):
        return self.__crews_poisoned
    
    def __updateCrewsIsFreezed(self):  #依照隊友數量更新石化狀態
        if not self.__ingame:
            return
        self.__crews_freezed = []
        for i in range(self.__crews):
            crew_freezed_img = self.__screenshot[235+(i*75):280+(i*75), 285:325]
            crew_freezed = self.__freezed.templateMatching(crew_freezed_img, 0.8)
            if len(crew_freezed) > 0:
                self.__crews_freezed.append(1)
            else:
                self._crews_freezed.append(0)
        return 
    
    def getCrewsIsFreezed(self):
        return self.__crews_freezed

if __name__ == '__main__':
    d = Detector(0)
    d.start()
    time.sleep(5)
    d.stop()

    d = Detector(1)
    d.start()
    time.sleep(5)
    d.stop()
