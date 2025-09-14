import os
import sqlite3
import pandas as pd
import numpy as np


from flask import Blueprint, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug.utils import secure_filename
#from nyancheck.net.predict import predict

import glob
import random
import shutil
import re

selectimage_bp = Blueprint('selectimage', __name__, template_folder='templates', static_folder="./static", static_url_path="/static")

upload_dir = './uploads'
allowed_extensions = set(['png', 'jpg', 'gif'])
config = {}
config['upload_dir'] = upload_dir


# index.htmlの「画像選択」ボタン）がクリックされたときの処理
#@selectimage_bp.route('/api/v1/randomselect', methods=['GET', 'POST'])
def randomselect():
   #if request.method == 'POST':

   list=glob.glob('/Users/iikubo/Library/CloudStorage/OneDrive-KyushuUniversity/2_work/open_campus/battle/nyancheck/net/data/validation_data/*/*.jpg')
   #print(list)

   filenames = []
   ans = []
   img_url_n = []
   nyan_types = []

   shutil.rmtree(upload_dir)
   os.mkdir(upload_dir)

   #空のデータフレーム
   df_image = pd.DataFrame()

   for i in range(10):
      data=random.choice(list)

      #for image
      filename=os.path.split(data)[1]
      filenames.append(filename)
      #shutil.copyfile(data, data1)

      #for answer
      subdirname = os.path.basename(os.path.dirname(data))
      ans.append(subdirname)

      shutil.copy(data, upload_dir)
      img_url_n.append('/uploads/' + filename)
      #nyan_types.append(predict(filename))

      list_=[[i,filename,subdirname]]
      df_new = pd.DataFrame(list_,columns=["ID", "filename", "猫の種類"])
      df_image = pd.concat([df_image,df_new])

   df_image.set_index("ID", inplace = True)
   df_image=df_image.replace({"Abyssinian":"アビシニアン","american shorthair":"アメリカンショートヘアー","Dog":"犬","Egyptian Mau":"エジプシャンマウ","japanese cat":"日本猫","Maine Coon":"メインクーン","Norwegian Forest Cat":"ノルウェージャンフォレストキャット","Russian Blue":"ロシアンブルー"})


   #df_image = pd.DataFrame(zippedList,
   #                        columns=["filename", "answer"],
   #                        index=["ID1", "ID2", "ID3", "ID4", "ID5", "ID6", "ID7", "ID8", "ID9", "ID10"])


   #return render_template('index.html', img_url_n=img_url_n, filenames=filenames, anss=ans, nyan_types=nyan_types)
   #return render_template('index.html', img_url_n=img_url_n, filenames=filenames, anss=ans)
   return df_image


   #else:
   #   return redirect(url_for(''))

