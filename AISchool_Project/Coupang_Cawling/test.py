#from src.crawl_with_page import Coupang
from src.crawl_no_page import Coupang
# 상품 목록 페이지의 URL



coupang = Coupang()
URL = coupang.input_review_url()
#print(coupang.get_product_code(coupang.get_product_links(product_list_link=URL)))
a = coupang.get_product_links(product_list_link=URL)
# for i in a:
#     print(i)
#     print()
print(a)
# print(len(a))