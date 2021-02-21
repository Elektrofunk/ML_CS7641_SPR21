import os
import sys
from datetime import datetime
from pathlib import Path
class Report:

    def __init__(self,name):
        now = datetime.now()
        self.init_time =now.strftime("%H_%M_%S")
        self.date = now.strftime("%Y%m%d")
        self.cwd = os.getcwd()


        if sys.platform.startswith('linux'):
            self.div = "/"
        else:
            self.div = "\\"
        self.daily_dir = self.cwd+self.div+self.date
        if not os.path.exists(self.daily_dir):
            os.makedirs(self.daily_dir)

        self.run_dir = self.daily_dir + self.div + name+'_'+ self.init_time
        Path(self.run_dir).mkdir(parents=True, exist_ok=True)

    def get_time(self):
        return self.init_time
    def get_path_and_prefix(self):
        return self.run_dir + self.div+self.time + "_"

    #Source: https: // stackoverflow.com / questions / 23556040 / moving - specific - file - types - with-python
    def move_report_files(self):
        import os
        import shutil
        sourcepath = self.cwd
        sourcefiles = os.listdir(sourcepath)
        destinationpath = self.run_dir
        for file in sourcefiles:
            if file.endswith('.png')  or file.endswith('.csv') or file.endswith('.pickle') :
                shutil.move(os.path.join(sourcepath, file), os.path.join(destinationpath, file))
