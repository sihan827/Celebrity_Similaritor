from urllib.request import urlretrieve
from urllib.parse import quote_plus
from bs4 import BeautifulSoup as Bs
from selenium import webdriver
import shutil
import os


def get_keyword_list(path):
    keywords = []
    try:
        # Be careful about encoding
        fin = open(path, 'rt', encoding='utf-8')
        while True:
            line = fin.readline()
            if not line:
                break
            keywords.append(line[:-1])
        fin.close()
        return keywords
    except FileNotFoundError:
        print('No txt file in path')
        return -1


def search_img(keyword, gender, limit):
    url = f'https://www.google.com/search?q={quote_plus(keyword)}' \
          f'&sxsrf=ALeKk02jZiNpixyPIOho-HXr4GMeXTQoyw:1612103615267' \
          f'&source=lnms&tbm=isch' \
          f'&sa=X&ved=2ahUKEwjbx9D6scbuAhVTUd4KHdmcASYQ_AUoAXoECAQQAw' \
          f'&biw=890&bih=957'
    browser = webdriver.Chrome('C:/chromedriver.exe')
    browser.get(url)

    while len(browser.find_elements_by_tag_name('img')) < limit:
        browser.execute_script('window.scrollTo(0, document.body.scrollHeight)')
        browser.implicitly_wait(1)

    img_count = len(browser.find_elements_by_tag_name('img'))
    print('Image Loaded:', img_count)

    html = browser.page_source
    soup = Bs(html, 'html.parser')
    img = soup.select('.rg_i.Q4LuWd')

    dir = './' + str(gender) + '/' + str(keyword)
    if not os.path.exists(dir):
        os.makedirs(dir)

    n = 1
    for i in img:
        try:
            img_src = i.attrs['src']
            urlretrieve(img_src, dir + '/' + keyword + '_' + str(n) + '.png')
            n += 1
        except KeyError:
            img_src = i.attrs['data-src']
            urlretrieve(img_src, dir + '/' + keyword + '_' + str(n) + '.png')
            n += 1

    browser.close()


def make_utk_dir():
    data_path = './UTKFace'
    target_path = './utk_face'
    if not os.path.exists(data_path):
        print('No UTKFace directory!')
        return
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    file_list = os.listdir(data_path)
    for file in file_list:
        age = file.split('_')[0]
        age_path = target_path + '/' + str(age)
        if not os.path.exists(age_path):
            os.makedirs(age_path)
        src = data_path + '/' + str(file)
        dest = age_path + '/' + str(file)
        shutil.move(src, dest)


if __name__ == '__main__':
    # filepath = './female_celebrities.txt'
    # kws = get_keyword_list(filepath)
    # for kw in kws:
    #     print(kw)

    # search_img('이영호', 'male', LIMIT)

    # make_utk_dir()
    LIMIT = 150

    # Female Celebrity Crawling
    filepath = '../female_celebrities.txt'
    kws = get_keyword_list(filepath)
    for kw in kws:
        search_img(kw, 'female', LIMIT)

    # Male Celebrity Crawling
    filepath = '../male_celebrities.txt'
    kws = get_keyword_list(filepath)
    for kw in kws:
        search_img(kw, 'male', LIMIT)
