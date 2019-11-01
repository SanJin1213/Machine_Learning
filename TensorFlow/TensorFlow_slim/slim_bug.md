# slim中遇见的问题及解决办法

## 1. 写脚本文件的问题

### 1.下载flowers及转换成tfrecord文件的问题

```shell
$ DATA_DIR=/tmp/data/flowers
$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir="${DATA_DIR}"
```

1）复制时，不要复制$这符号，会导致 command not found
2) 如果显示no such file or directory,可以在脚本文件添加cd 到文件目录文件夹命令
3）使用自己新建的docker，这种方式不建议，因为新建docker就需要重新配置坏境
4）也可以简写成

```shell
python download_and_convert_data.py --dataset_name=flowers --dataset_dir=文件的位置
```
