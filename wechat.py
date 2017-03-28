import wxpy
bot = wxpy.Bot(console_qr=True)
logger = wxpy.get_wechat_logger(receiver=bot, name='word2vec')
logger.warning('这是一条 WARNING 等级的日志，你收到了吗？')