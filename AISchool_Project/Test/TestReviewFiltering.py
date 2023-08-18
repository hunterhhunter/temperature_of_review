from AISchool_Project.Function.ReviewFiltering import FilterReviewsManager
from AISchool_Project.Function.ReviewFiltering import show_predefined_filtering, show_user_input_filtering

# 예시 데이터 경로 딕셔너리
data_paths = {
    "data1": r"C:\Users\gjaischool\Documents\GitHub\-\AISchool_Project\modeling(look_this)\data\data_with_prodname_phone.csv",
    "data2": r"C:\Users\gjaischool\Documents\GitHub\-\AISchool_Project\Test\data\data_with_prodname_ketboard.csv",
    # ...
}

# 클래스 인스턴스 생성
manager = FilterReviewsManager()

# 여러 DataFrame 로드
manager.load_dataframes(data_paths)

# 첫 번째 데이터에 대한 사용자 입력 필터링
key_for_user_filter = "data1"  # 예시 키
show_user_input_filtering(manager, key_for_user_filter)

# 지정된 문자열 목록을 이용한 필터링(광고성 리뷰 제거)
strings_to_filter = ['쿠팡체험이벤트단', '쿠팡체험단', '쿠팡 체험 이벤트', '쿠팡 및 쿠팡의 계열회사 직원이 상품을 제공 받아 작성한 후기입니다.', '쿠팡무료체험단']
show_predefined_filtering(manager, key_for_user_filter, strings_to_filter)