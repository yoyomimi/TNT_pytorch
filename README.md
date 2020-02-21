# TNT_pytorch
This is a PyTorch version for MOT TrackletNet

- implementation timeline http://note.youdao.com/noteshare?id=fde3925cdd1b82d39a4124c961737c8e

## detection

### data dir structure

-data/ -training/ -objtrack/ -images/ -0000/ -0000xx.png

                                      -0001/ -0000xx.png
                                      
                             -labels/ -0000.txt
                             
                                      -0001.txt
                                      
       -eval/     -objtrack/ -images/ -0000/ -0000xx.png
       
                                      -0001/ -0000xx.png
                                      
                             -labels/ -0000.txt
                             
                                      -0001.txt
                                      
e.g. data/training/objtrack/images/0000/000000.png
     data/training/objtrack/labels/0000.txt

### detection run

see detection/detect.sh

### environments

see requirements.txt

pip install -r requirements.txt
       
                    
