import pandas as pd

class FilterReviewsManager:
    def __init__(self):
        self.dataframes = {}

    def load_dataframes(self, data_paths: dict):
        """여러 DataFrame들을 딕셔너리 형식으로 로드합니다."""

        for key, path in data_paths.items():
            self.dataframes[key] = pd.read_csv(path, encoding='cp949', index_col=0)

    def extract_filter_strings(self, key: str, strings_to_filter: list) -> pd.DataFrame:
        """지정된 문자열이 포함되지 않는 행을 반환합니다."""

        df = self.dataframes[key].copy()
        for string in strings_to_filter:
            df = df[~df['리뷰'].str.contains(string, na=False, case=False)]
        return df

    def __extract_rows_with_strings(self, key: str, strings_to_extract: list) -> pd.DataFrame:
        """지정된 문자열이 포함된 행을 반환합니다."""

        extracted_df = self.dataframes[key].copy()
        for string in strings_to_extract:
            extracted_df = extracted_df[extracted_df['리뷰'].str.contains(string, na=False, case=False)]
        return extracted_df

    def get_dataframe(self, key: str) -> pd.DataFrame:
        """지정된 키의 DataFrame을 반환합니다."""

        return self.dataframes.get(key, None)

    def clear_dataframe(self, key: str):
        """지정된 키의 DataFrame을 초기화합니다."""

        if key in self.dataframes:
            del self.dataframes[key]

    def get_input_user_keyword_row(self, key: str) -> pd.DataFrame:
        """사용자 입력을 받아 해당 문자열이 포함된 행을 반환합니다."""

        user_input = input("뽑아낼 키워드를 띄어쓰기로 구분하여 입력하세요: ")
        strings_to_extract = user_input.split()
        return self.__extract_rows_with_strings(key, strings_to_extract)



def show_user_input_filtering(manager: FilterReviewsManager(), key_for_user_filter: str):
    """사용자가 입력한 키워드 결과가 있는 행들 보여준다. """
    if key_for_user_filter in manager.dataframes:
        filtered_df_by_user = manager.get_input_user_keyword_row(key_for_user_filter)
        print("사용자 입력에 따른 필터링 결과:")
        print(filtered_df_by_user)


def show_predefined_filtering(manager: FilterReviewsManager(), key_for_user_filter: str, strings_to_filter: list):
    """광고성 리뷰들을 제거하고 남은 행들을 보여준다. """
    if key_for_user_filter in manager.dataframes:
        filtered_df_mod = manager.extract_filter_strings(key_for_user_filter, strings_to_filter)
        print(f"\n지정된 문자열 목록에 따른 {key_for_user_filter} 데이터 필터링 결과:")
        print(filtered_df_mod)


