import config
config.USE_ESP32_CAMERA = False
import web_ui
web_ui.USE_ESP32_CAMERA = False
web_ui.main()
