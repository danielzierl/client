import sys
import webbrowser
import time
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from pynput.keyboard import Key
from selenium.webdriver.chrome.service import Service


import time


class pdfReader:

    def on_key_release(actions, key):

            if key == 1:
                actions.send_keys(Keys.ARROW_RIGHT)
                actions.perform()
            elif key == 2:
                actions.send_keys(Keys.ARROW_LEFT)
                actions.perform()
            elif key == 3:
                print("Up key clicked")
            elif key == 4:
                print("Down key clicked")
            elif key == Key.esc:
                exit()




