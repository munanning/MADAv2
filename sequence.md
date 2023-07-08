# The sequence of execution

## Active learning with Fast iteration

1. Step 1 (source):
* get the source features
~~~~
python3 step1_save_feat_source.py
~~~~

* cluster the source anchors
~~~
python3 step1_cluster_anchors_source.py: 
~~~

* select active samples with ratio of 0.01
~~~
python3 step1_select_active_samples.py: 
~~~

* train stage1 model
~~~
python3 step1_train_active_suponly.py
~~~

2. Step 2 (iterate for n times in target):
* get the target features
~~~~
python3 step2_n_save_feat_target.py
~~~~

* cluster the target anchors
~~~
python3 step2_n_cluster_anchors_source.py: 
~~~

* additionally select active samples
~~~
python3 step2_n_select_active_samples.py: 
~~~

* train model
~~~
python3 step1_train_active_suponly.py
~~~
