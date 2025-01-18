# from importlib import import_module 
# from wrangling_scripts.wrangle_metrics import return_metrics


# run_script = import_module('app.run')

# if __name__ == '__main__':
#     run_script.main()
 
    
from app.run import app
import sys 
sys.path.append('../')



app.run(host='0.0.0.0', port=3000, debug=True)    
    