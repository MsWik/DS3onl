import multiprocessing
import time
import os


try: 
    os.mkdir('ImagesMulti')
except FileExistsError:
    shutil.rmtree ('ImagesMulti')
    os.mkdir ('ImagesMulti')
    
list_url = ["https://lenta.ru/"] 
images = []
for url in list_url:
    response = requests.get(list_url)
    soup = BeautifulSoup(response.text, features="html.parser")
    for img in soup.findAll('img'):
        images.append(img.get('src'))
        
img_list_cleaned=[]
for i in range(len(images)):
    if images[i]==None or images[i].startswith('//'):
        continue
    elif images[i].endswith('jpg'):
        img_list_cleaned.append(images[i])
        
def img_downl(url,pic_num,dir_name):
    # Начинаем загрузку.
        file_path=dir_name+'\\image'+str(pic_num)+'.jpg'
        img=requests.get(url)
        open(file_path,"wb").write(img.content)

if __name__=='__main__':
    start = time.time()
    processes=[]
    for i in img_list_cleaned:
        p=multiprocessing.Process(target=img_downlo, args=[i, img_list_cleaned.index(i),'ImagesMulti'])
        p.start()
    for p in processes:
        p.join()   
    
    end = time.time()
    print('Time taken in seconds -', end - start)