# for i in "1" "2" "3" "4"
# do
python3 eval/benchmark_gemini.py -file_name uncertainty_mean_ucb_0_10000_10500_greedy_442.json  -iter_index 0 -mode uncertainty_mean_ucb_greedy_442
# done

# for i in "1" "2" "3" "4"
# do
python3 eval/benchmark_gemini.py -file_name uncertainty_rand_0_10000_10500_greedy_442.json  -iter_index 0 -mode uncertainty_rand_greedy_442
# done

# for i in "1" "2" "3" "4"
# do
python3 eval/benchmark_gemini.py -file_name rand_None_0_10000_10500_greedy_442.json  -iter_index 0 -mode rand_None_greedy_442
# done

# for i in "1" "2" "3" "4"
# do
python3 eval/benchmark_gemini.py -file_name uncertainty_lcb_0_10000_10500_greedy_442.json  -iter_index 0 -mode uncertainty_lcb_greedy_442
# done

# for i in "1" "2" "3" "4"
# do
python3 eval/benchmark_gemini.py -file_name uncertainty_sec_ucb_0_10000_10500_greedy_442.json  -iter_index 0 -mode uncertainty_sec_ucb_greedy_442
# done