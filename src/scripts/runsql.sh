#!/bin/bash


total=`grep -c '^'<$1`
step=10000
it=$(expr $total / $step)
for i in $(seq 1 1 "$it"); do
h=$(expr $i \* $step)
echo "head -$h $1|tail -$step > tmp.sql"
head -$h $1|tail -$step > tmp.sql
sqlcmd -S localhost -U SA -P 'YOUR_PASS_WORD' -d your_db -I -i tmp.sql

done

last=$(expr $total - $it \* $step)
echo "tail -$last $1> tmp.sql"
tail -$last $1> tmp.sql
sqlcmd -S localhost -U SA -P '$YOUR_PASS_WORD' -d your_db -I -i tmp.sql
rm -rf tmp.sql

