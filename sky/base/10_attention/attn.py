# 不随意：不追随自己意志，比如一眼看到灰色物品中的红色被子
# 随意：指的是追随自己的意志有意识的，是指看完杯子后可能会留意到书本

# conv、fc、pool只考虑不随意线索
# query就是随意线索（自己的想法）
# 直接平均->加权平均，给定一个x根据候选x和实际x的距离加权
