migrantion_detector
======
**migrantion_detector** is a Python toolbox to detect migration events in digital trace data, such as Call Detail Record (CDR), geo-tagged tweets, and other check-in data. It is able to handle a large volume of data (TB) and provides useful functions such as plotting the trajectory of a migrant and export your results.

<kbd><img src="example/1_90.png" /></kbd>

How to install it
------
`pip install migrantion_detector`

> **_NOTE:_**
- migrantion_detector has a dependency on [turi/GraphLab](https://turi.com/) to speed up the computation by parallel computing (It only took about 40 minutes to detect migrants using 600 million unique trajectory records.). You need to apply for a [license](https://turi.com/download/academic.html) and [install](https://turi.com/download/install-graphlab-create.html) it before installing migrantion_detector.
- It is recommended to create a new Python 2.7 environment to install **GraphLab** and **migrantion_detector**.
- Other requires: pandas, numpy, matplotlib, and seaborn.

How to use it
------
migrantion_detector is easy to use, just like pandas. First, you need to import your trajectory dataset and then detect the migrants.
```
import migration_detector as md

traj = md.read_csv('example/migrant_location_history_example1.csv')

migrants = traj.find_migrants()
print(migrants)

# plot a migrant's trajectory
traj.plot_migration_segment(migrants[0])

# save the result of detected migrants
md.to_csv(migrants, 'migrants_result.csv')
```

Format of the input trajectory data
------
The input file should contain at least three columns: user_id(`int` or `str`), date(`YYYYMMDD`), location_id(`int` or `str`). The *location* depends on the definition of the migration, such as district, state, or city. Here is an example of trajectory data.

| user_id | date     | location |
|---------|----------|----------|
| 1       | 20180701 | 1        |
| 1       | 20180701 | 2        |
| 1       | 20180702 | 1        |


How to cite it
------
TO ADD
