import pandas as pd

# (1) CSV 혹은 Excel에서 읽어오기 (파일 경로와 함수는 상황에 맞춰 수정)
df = pd.read_csv('raw_data.csv')

# (2) 생산량(Kg) 쉼표 제거 후 정수 변환
df['생산량(Kg)'] = (
    df['생산량(Kg)']
    .astype(str)
    .str.replace(',', '')
    .fillna('0')
    .replace('', '0')
    .astype(int)
)

# (3) 1~10차 출고량 각각 쉼표 제거 후 정수 변환
for i in range(1, 11):
    qty_col = f'{i}차 출고량'
    if qty_col in df.columns:
        df[qty_col] = (
            df[qty_col]
            .astype(str)
            .str.replace(',', '')
            .fillna('0')
            .replace('', '0')
            .astype(int)
        )

# (4) 날짜 컬럼들(생산일자, 시작, 종료, n차 출고날짜)을 모두 datetime 변환 후 date만 추출
df['생산일자'] = pd.to_datetime(df['생산일자'], errors='coerce').dt.date
df['시작'] = pd.to_datetime(df['시작'], errors='coerce').dt.date
df['종료'] = pd.to_datetime(df['종료'], errors='coerce').dt.date

shipping_date_cols = [f'{i}차 출고날짜' for i in range(1, 11)]
for col in shipping_date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce').dt.date

# (5) 일자별 생산량 합산 (생산일자 기준)
production_daily = df.groupby('생산일자')['생산량(Kg)'].sum().reset_index()
production_daily.columns = ['date', 'produce']  # 컬럼명 변경

# (6) 일자별 출고량 합산
# 각 행에서 1~10차 출고날짜와 출고량을 추출하여 새로운 리스트 생성
shipping_data_list = []
for _, row in df.iterrows():
    for i in range(1, 11):
        date_col = f'{i}차 출고날짜'
        qty_col = f'{i}차 출고량'
        if date_col in df.columns and qty_col in df.columns:
            if pd.notnull(row[date_col]) and row[qty_col] != 0:
                shipping_data_list.append({
                    'date': row[date_col],
                    'release': row[qty_col]
                })

# shipping_data_list를 DataFrame으로 만들어서 날짜별 출고량 합산
shipping_df = pd.DataFrame(shipping_data_list)
shipping_daily = shipping_df.groupby('date')['release'].sum().reset_index()

# (7) production_daily 와 shipping_daily 를 날짜(date) 기준으로 병합(outer join)
final_df = pd.merge(production_daily, shipping_daily, on='date', how='outer')

# 생산량이나 출고량이 없는 날짜에 대해서는 0으로 채움
final_df['produce'] = final_df['produce'].fillna(0).astype(int)
final_df['release'] = final_df['release'].fillna(0).astype(int)

# (8) 최종 CSV로 저장
final_df.to_csv('data.csv', index=False)

print(final_df)
