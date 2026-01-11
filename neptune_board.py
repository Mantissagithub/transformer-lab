import neptune
from neptune_tensorboard import enable_tensorboard_logging

run = neptune.init_run(
    project="mantissa6789/hyperconnections",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0YTU2NTU1My0xMjJlLTQ4ZTgtYjI2ZS1mNjlkZjg0ZDY3NGUifQ==",
)

enable_tensorboard_logging(run, log_dir="./runs")
run.stop()
