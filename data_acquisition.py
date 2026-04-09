import os
from dotenv import load_dotenv
import shutil
import kagglehub

class Data_acquisition:
    def __init__(self):
        self.api = self.connect_kaggle_api()
        csv_files = self.fetch_dataset_file_names()
        self.dict_of_weeks = self.group_by_weeks(csv_files)

    # Connects to Kaggle API with environment variable credentials
    def connect_kaggle_api(self):
        load_dotenv(override=True)
        os.environ['KAGGLE_KEY'] = os.environ['KAGGLE_API_TOKEN']
        import kaggle
        api = kaggle.api
        return api

    # Fetches Dataset .CSV File Names
    def fetch_dataset_file_names(self) -> list[str]:
        all_files = []
        page_token = None
        while True:
            response = self.api.dataset_list_files("zilinpeng/cesnet-quic22", page_token=page_token, page_size=100)
            all_files.extend(response.files)
            page_token = response.nextPageToken
            if not page_token:
                break

        csv_files = [f.name for f in all_files if f.name.endswith(".csv")]
        return csv_files

    # Groups into a dictionary where values are the file names and the key is the week relating to those files
    def group_by_weeks(self, file_names: list[str]) -> dict:
        dict_of_weeks = {'week_1': [i for i in file_names if "W-2022-44" in i],
                         'week_2': [i for i in file_names if "W-2022-45" in i],
                         'week_3': [i for i in file_names if "W-2022-46" in i],
                         'week_4': [i for i in file_names if "W-2022-47" in i]}

        dict_of_weeks['week_1'] = [dict_of_weeks['week_1'][0]]
        dict_of_weeks['week_2'] = [dict_of_weeks['week_2'][0]]
        dict_of_weeks['week_3'] = [dict_of_weeks['week_3'][0]]
        dict_of_weeks['week_4'] = [dict_of_weeks['week_4'][0]]

        return dict_of_weeks

    # Downloads datasets from KaggleHub
    def download_datasets(self, weeks_to_download: list[str]) -> None:
        dataset_root = './dataset'
        download_tmp = './dataset_temp'
        duckdb_root = './duckdb'
        days_of_week = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

        for key, value in self.dict_of_weeks.items():
            os.makedirs(os.path.join(dataset_root, key), exist_ok=True) #Makes ./dataset folder
            os.makedirs(os.path.join(duckdb_root, key), exist_ok=True)  #Makes ./duckdb folder

            if key in weeks_to_download:
                for day_iterator, day_file in enumerate(value):
                    print(f'Downloading: {day_file}')
                    path = kagglehub.dataset_download(
                        "zilinpeng/cesnet-quic22",
                        force_download=False,
                        output_dir=download_tmp,
                        path=day_file
                    )

                    current_day_of_week = days_of_week[day_iterator]
                    self.move_datasets_and_rename(download_tmp, dataset_root, key, current_day_of_week)

        shutil.rmtree(download_tmp)
        print(f'Download complete at for {len(weeks_to_download)} weeks in {dataset_root}')

    # Helper method which moves datasets to flatten directory and renames .CSV files
    def move_datasets_and_rename(self, download_tmp, dataset_root, key, current_day_of_week):
        src_dir = os.path.join(download_tmp, 'cesnet-quic22')
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                src = os.path.join(root, file)
                ext = os.path.splitext(file)[1]
                dest = os.path.join(dataset_root, key, f"{current_day_of_week}.csv")
                shutil.move(src, dest)