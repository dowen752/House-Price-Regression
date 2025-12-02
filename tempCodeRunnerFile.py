numeric_data = data.select_dtypes(include=[np.number, np.float64, np.int64])
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    # plt.show()