import re
import time

class Helper(object):

    @staticmethod
    def scroll_down(driver, scroll):
        a = 0
        b = 2000

        for i in range(scroll):
            # Scroll down to bottom
            driver.execute_script("window.scrollTo({}, {});".format(a, b))
            a = b
            b += 2000

            # implicity
            driver.implicitly_wait(10)
            # Wait to load page
            time.sleep(10)


    @staticmethod
    def toDate(input):
        re_date = r"(\d{1,2}) ([A-Za-z]+).? (\d{4})"
        output = re.search(re_date, input)
        date = output.group(1)
        month = output.group(2)
        year = output.group(3)

        month = month.lower()
        months = ['januari', 'februari', 'maret', 'april', 'mei', 'juni', 'juli', 'agustus', 'september', 'oktober',
                  'november', 'desember']
        months_en = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                     'november', 'december']

        if months.count(month) > 0:
            month = months.index(month) + 1
        elif months_en.count(month) > 0:
            month = months_en.index(month) + 1

        return str(year) + "-" + str(month) + "-" + str(date)

    @staticmethod
    def getNumber(input):
        output = input.lower().replace(' comments','').replace(' shares', '')
        return output
