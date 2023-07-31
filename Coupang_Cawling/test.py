from crawl_with_page import get_product_links
from crawl_with_page import Coupang
# 상품 목록 페이지의 URL



coupang = Coupang()
URL = coupang.input_review_url()
print(coupang.get_product_code(get_product_links(product_list_link=URL)))
# print(get_product_links(product_list_link=URL))