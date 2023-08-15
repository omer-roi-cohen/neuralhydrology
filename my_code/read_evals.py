import pandas as pd
from pathlib import Path
from graphs_utils import plot_nse_cdf

run_dir = Path(r'runs\cudalstm_531_default_lstm_10_epochs')
output_filer_path = run_dir / "test/model_epoch010/test_metrics.csv"
# load test results of the base run
df_final = pd.read_csv(output_filer_path, dtype={'basin': str})
df_final = df_final.set_index("basin")

print(f"Median NSE of the test period {df_final['NSE'].median():.3f}")

plot_nse_cdf(output_filer_path, 'NSE_CDF_graph.png')
