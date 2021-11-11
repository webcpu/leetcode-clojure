(ns leetcode.core
  (:gen-class)
  (:require [clojure.string :as str]
            [clojure.set :as set]))

;; ;;1
;; (defn two-sum [nums target]
;;   (let [check (fn [[result index-map] index]
;;                 (let [num (get nums index)
;;                       index2 (get index-map (nums index))]
;;                   (if (nil? index2)
;;                     [[] (assoc index-map (- target num) index)]
;;                     (reduced [[index2 index] index-map]))))]
;;   (first (reduce check [[] {}] (range (count nums))))))
;; (map (partial apply two-sum) ['([2 7 11 15] 9) '([3 2 4] 6) '([3 3] 6)])

;; ;;9
;; (defn is-palindrome [x]
;;   (let [palindrome? (fn [x]
;;                       (= (vec (str x))
;;                          (reverse (vec (str x)))))]
;;     (cond
;;       (neg? x) false
;;       (zero? x) true
;;       :else (palindrome? x))))
;; (map is-palindrome [121 -121 10 -101])
;; ;;(map is-palindrome ["A man, a plan, a canal: Panama", "race a car"])

;; ;;13
;; (defn roman-to-int [s]
;;   (let [roman-map {\I 1 \V 5 \X 10 \L 50 \C 100 \D 500 \M 1000}
;;         add (fn [[result prev] num]
;;               (if (> prev num)
;;                 [(- result num) prev]
;;                 [(+ result num) num]))
;;         digits (map #(get roman-map %) (reverse s))]
;;    (first (reduce add [0 0] digits))))
;; (map roman-to-int ["III" "IV" "IX" "LVIII" "MCMXCIV"])

;; ;;14
;; (defn longest-common-prefix [strs]
;;   (let [min-len (apply min (map count strs))
;;         compare-char (fn [len i]
;;                        (if (apply = (map #(subs % i (inc i)) strs))
;;                          (inc len)
;;                          (reduced len)))]
;;     (subs (first strs) 0 (reduce compare-char 0 (range min-len)))))
;; (map longest-common-prefix [["flower" "flow" "flight"] ["dog" "racecar" "car"]])

;; ;;20
;; (defn is-valid [s]
;;   (let [cs (vec s)
;;         bracket-map {\{ \} \( \) \[ \] }
;;         add (fn [result c]
;;               (cond
;;                 (empty? result) [c]
;;                 (= c (get bracket-map (last result))) (drop-last result)
;;                 :else (conj result c)))
;;         ]
;;     (->> (reduce add [(first cs)] (rest cs))
;;          (empty?))))
;; (map is-valid ["()" "()[]{}" "(]"  "([)]" "{[]}"])

;; ;;26
;; (defn remove-duplicates [nums]
;;   (let [len (count nums)
;;         xs (into-array nums)
;;         move (fn [index_of_uniqs i]
;;                (cond
;;                  (= (inc i) len)
;;                  (do
;;                    (when (= (inc i) len)
;;                        (aset xs (inc index_of_uniqs) (aget xs i)))
;;                      index_of_uniqs)
;;                  (not= (aget xs (dec i)) (aget xs i)) (inc index_of_uniqs)
;;                  :else (do
;;                          (aset xs index_of_uniqs (aget xs i))
;;                          index_of_uniqs)))]
;;     (reduce move 0 (range 1 len))
;;     (vec xs)))

;; (map remove-duplicates [[1 1 2] [0 0 1 1 1 2 2 3 3 4]])

;; ;;27
;; (defn remove-element [nums val]
;;   (let [xs (into-array nums)
;;         remove-val (fn [offset i]
;;                      (let [num (aget xs i)]
;;                        (if (= val num)
;;                          (inc offset)
;;                          (do
;;                            (aset xs (- i offset) num)
;;                            offset))))]
;;     (reduce remove-val 0 (range 0 (count xs)))
;;     (vec xs)))
;; (map (partial apply remove-element) ['([3 2 2 3] 3) '([0 1 2 2 3 0 4 2] 2)])

;; ;;28 ;;HARD
;; (defn str-str1 [haystack needle]
;;   (letfn [(str-str' [haystack needle]
;;             (let [cols (count haystack)
;;                   rows (count needle)
;;                   dp (make-array Long/TYPE rows cols)
;;                   indices (for [row (range rows) col (range cols)]
;;                             [row col])
;;                   cs1 (vec haystack)
;;                   cs2 (vec needle)
;;                   calculate-score (fn [row col]
;;                                     (if (= (cs2 row) (cs1 col))
;;                                       1
;;                                       0))]
;;               (doseq [[row col] indices]
;;                 (aset dp row col (calculate-score row col)))
;;               (mapv vec dp)))]
;;     (if (empty? (vec needle))
;;       0
;;       (str-str' haystack needle))))

;; (defn str-str [haystack needle]
;;   (let [len (count needle)
;;         first-letter (first (vec needle))
;;         hs (vec haystack)
;;         indices (reduce (fn [result index]
;;                           (if (= (hs index) first-letter)
;;                             (conj result index)
;;                             result)) [] (range (count haystack)))
;;         indexof (fn [result index] (if (= (subs haystack index (+ index len)) needle)
;;                                      (reduced index)
;;                                      result))]
;;     (if (empty? (vec needle))
;;       0
;;       (reduce indexof -1 indices))))
;; (map (partial apply str-str) ['("hello" "ll") '("aaaaa" "bba") '("" "")])

;; ;;35
;; (defn search-insert [nums target]
;;   (letfn [(search [nums target start end]
;;             (let [middle (quot (+ start end) 2)]
;;               (cond
;;                 (> start end) start
;;                 (= (nums middle) target) middle
;;                 (> (nums middle) target) (search nums target start (dec middle))
;;                 (< (nums middle) target) (search nums target (inc middle) end))))]
;;     (search nums target 0 (dec (count nums)))))
;; (map (partial apply search-insert) ['([1 3 5 6] 5) '([1 3 5 6] 2) '([1 3 5 6] 7) '([1 3 5 6] 0) '([1] 0)])

;; ;;53
;; (defn max-sub-array [nums]
;;   (let [indexed-positives (filter #(pos? (last %)) (map vector (range (count nums)) nums))
;;         indices (map first indexed-positives)
;;         check (fn [[max-sum sum start] index]
;;                 (let [num (nums index)]
;;                   (if (and (<= sum 0) (pos? num))
;;                     [(max max-sum num) num index]
;;                     [(max max-sum (+ sum num)) (+ sum num) start])))

;;         max-sub-array' (fn [nums indices]
;;                          (let [left (first indices)
;;                                sum (nums left)]
;;                          (first (reduce check [sum sum left] (range (inc left) (inc (last indices)))))))]

;;     (if (< (count indices) 2)
;;       (apply max nums)
;;       (max-sub-array' nums indices))))
;; (map max-sub-array [[-2,1,-3,4,-1,2,1,-5,4] [1] [5,4,-1,7,8]])

;; ;;58
;; (defn length-of-last-word [s]
;;   (->> (str/trim s)
;;        (re-seq #"\w+")
;;        (last)
;;        (count)))
;; (map length-of-last-word ["Hello World" "   fly me   to   the moon  " "luffy is still joyboy"])

;; ;;66
;; (defn plus-one [digits]
;;   (let [add (fn [[results carry] digit]
;;               (let [result (+ digit carry)]
;;                 [(cons (rem result 10) results) (quot result 10)]))
;;         [results carry] (reduce add [[] 1] (reverse digits))]
;;     (if (zero? carry)
;;       (vec results)
;;       (vec (cons carry results)))))
;; (map plus-one [[1 2 3] [4 3 2 1] [0] [9]])

;; ;;67
;; (defn add-binary [a b]
;;   (let [->digits (fn [s] (vec (reverse (map (fn [c] (- (int c) (int \0))) (vec s)))))
;;         ds1 (->digits a)
;;         ds2 (->digits b)
;;         len1 (count a)
;;         len2 (count b)
;;         len (max len1 len2)
;;         add (fn [[results carry] index]
;;               (let [digit1 (if (< index len1) (ds1 index) 0)
;;                     digit2 (if (< index len2) (ds2 index) 0)
;;                     result (+ digit1 digit2 carry)]
;;                 [(cons (rem result 2) results) (quot result 2)]))
;;         [digits carry] (reduce add [[] 0] (range len))
;;         result (if (zero? carry)
;;                  digits
;;                  (cons carry digits))]
;;     (apply str result)))
;; (map (partial apply add-binary) ['("11" "1") '("1010" "1011")])

;; ;;69
;; (defn my-sqrt [x]
;;   (letfn [(square [n] (* n n))
;;           (sqrt [x start end]
;;             (let [middle (quot (+ start end) 2)
;;                   product (square middle)
;;                   ]
;;               (cond
;;                 (> start end) end
;;                 (= product x) middle
;;                 (> product x) (sqrt x start (dec middle))
;;                 (< product x) (sqrt x (inc middle) end))
;;               )
;;             )]
;;     (sqrt x 0 (inc x))
;;     )
;;   )
;; (map my-sqrt [4 8 9 11 15 17])

;; ;;70
;; (defn climb-stairs [n]
;;   (let [climb (fn [n]
;;                 (let [climb' (fn [[a b] _]
;;                                [b (+ a b)])]
;;                   (last (reduce climb' [1 2] (range 2 n)))))]
;;     (case n
;;       1 1
;;       2 2
;;       (climb n))))
;; (map climb-stairs [2 3 4])

;; ;;88
;; ;; (defn merge' [nums1 m nums2 n]
;; ;;   (let [nums (into-array nums1)
;; ;;         merge-array' (fn [index i j]
;; ;;                        (let [a (nums1 i)
;; ;;                              b (nums2 j)]
;; ;;                          (cond
;; ;;                            (= a b) [(conj result a b) (inc i) (inc j)]
;; ;;                            (> a b) [(conj result b) i (inc j)]
;; ;;                            (< a b) [(conj result a) (inc i) j])))
;; ;;         merge-array (fn [[i j] _]
;; ;;                       (cond
;; ;;                         (= i m) (reduced (concat  (subvec nums2 j)))
;; ;;                         (= j n) (reduced (concat (subvec nums1 i)))
;; ;;                         :else (merge-array' i j)))]
;; ;;     (reduce merge-array [[] 0 0] (range (+ m n)))))
;; ;; (map (partial apply merge') ['([1 2 3 0 0 0] 3 [2 5 6] 3) '([1] 1 [] 0) '([0] 0 [1] 1)])

;; ;;118
;; (defn generate [num-rows]
;;   (let [generate-row (fn [results i]
;;                    (let [prev (last results)
;;                          len (dec (count prev))
;;                          add (fn [result index]
;;                                (conj result (+ (prev index) (prev (inc index)))))
;;                          result (conj (reduce add [1] (range len)) 1)]
;;                      (conj results result)))]
;;     (reduce generate-row [[1]] (range 1 num-rows))))
;; (map generate [5 1])

;; ;;119
;; (defn get-row [row-index]
;;   (let [get-row' (fn [prev index]
;;                    (let [add (fn [row i]
;;                                (let [num (+ (prev i) (prev (inc i)))]
;;                                  (conj row num)))
;;                          len (dec (count prev))
;;                          result (reduce add [1] (range 0 len))]
;;                      (conj result 1)))]
;;     (reduce get-row' [1] (range 0 row-index))))
;; (map get-row [3 0 1])

;; ;;121
;; (defn max-profit [prices]
;;   (let [make-profit (fn [[profit buy-price] price]
;;                       (if (< price buy-price)
;;                         [profit price]
;;                         [(max profit (- price buy-price)) buy-price]
;;                         )
;;                       )]
;;     (first (reduce make-profit [0 (first prices)] (rest prices)))))
;; (map max-profit [[7 1 5 3 6 4] [7 6 4 3 1]])

;; ;;125
;; (defn is-palindrome [s]
;;   (->> (filter #(Character/isLetterOrDigit %) (vec s))
;;        (map #(Character/toLowerCase %))
;;        (#(= % (reverse %)))
;;   ))
;; (map is-palindrome ["A man, a plan, a canal: Panama", "race a car"])

;; ;;136
;; (defn single-number [nums]
;;   (let [check-single-number (fn [num-map num]
;;                               (if (nil? (get num-map num))
;;                                 (assoc num-map num true)
;;                                 (dissoc num-map num)))]
;;    (first (keys (reduce check-single-number {} nums)))))
;; (map single-number [[2 2 1] [4 1 2 1 2] [1]])

;; ;;163
;; (defn find-missing-ranges [nums lower upper]
;;   (let [xs (vec (concat (concat [(dec lower)] nums) [(inc upper)]))
;;         ->range (fn [start end]
;;                   (if (= start end)
;;                     (str start)
;;                     (str start "->" end)))
;;         add (fn [result index]
;;               (let [num (xs index)
;;                     prev (xs (dec index))]
;;                 (if (= (inc prev) num) result
;;                   (conj result (->range (inc prev) (dec num))))))]
;;     (reduce add [] (range 1 (count xs)))))
;; (map (partial apply find-missing-ranges) ['([0 1 3 50 75] 0 99) '([] 1 1) '([] -3 -1) '([-1] -1 -1) '([-1] -2 -1)])

;; ;;167
;; (defn two-sum-ii [numbers target]
;;   (let [find-index (fn [[indices index-map] index]
;;                      (let [num (numbers index)
;;                            index2 (get index-map num)]
;;                        (if (nil? index2)
;;                           [indices (assoc index-map (- target num) (inc index))]
;;                           (reduced [[index2 (inc index)] index-map]))))]
;;    (first (reduce find-index [[] {}] (range (count numbers))))))
;; (map (partial apply two-sum-ii) ['([2,7,11,15] 9) '([2 3 4] 6) '([-1 0] -1)])

;; ;;168
;; (defn convert-to-title [column-number]
;;   (letfn [(convert [n]
;;             (cond
;;               (zero? n) []
;;               (< n 27) [n]
;;               (zero? (rem n 26)) (conj (convert (dec (quot n 26))) 26)
;;               :else (conj (convert (quot n 26)) (rem n 26))))
;;           ]
;;     (let [letters (vec "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
;;           ->letter (fn [index]
;;                      (letters (dec index)))]
;;       (->> (convert column-number)
;;            (map ->letter)
;;            (str/join "")))))
;; (map convert-to-title [1 52 28 701 2147483647])

;; ;;169
;; (defn majority-element [nums]
;;   (let [count-element (fn [num-map num]
;;                         (assoc num-map num (inc (or (get num-map num) 0))))
;;         num-map (reduce count-element {} nums)
;;         max-cnt (apply max (vals num-map))
;;         find-majority (fn [result [num cnt]]
;;                         (if (= cnt max-cnt)
;;                           (reduced num)
;;                           result))]
;;     (reduce find-majority nil num-map)))
;; (map majority-element [[3 2 3] [2,2,1,1,1,2,2]])

;; ;;171
;; (defn title-to-number [column-title]
;;   (let [letter-map (into {} (map vector (vec "ABCDEFGHIJKLMNOPQRSTUVWXYZ") (range 1 27)))
;;         ->number (fn [c] (get letter-map c))]
;;     (->> (map ->number (seq column-title))
;;          (reduce (fn [result num] (+ (* result 26) num)) 0))))
;; (map title-to-number ["A" "AB" "ZY" "FXSHRXW"])

;; ;;190
;; (defn reverse-bits [num]
;;   (letfn [(binary-digits' [n]
;;             (if (< n 2)
;;               [n]
;;               (conj (binary-digits' (bit-shift-right n 1)) (bit-and n 1))))
;;           (binary-digits [n]
;;             (if (zero? n)
;;               [0]
;;               (binary-digits' n)))]
;;     (let [digits (vec (reverse (binary-digits num)))
;;           new-digits (make-array Long/TYPE 32)]
;;       (doseq [index (range (count digits))]
;;         (aset new-digits index (digits index)))
;;       (reduce (fn [result digit] (+ (bit-shift-left result 1) digit)) 0 new-digits))))
;; (map reverse-bits [43261596 4294967293])

;; ;;191
;; (defn hamming-weight [num]
;;   (letfn [(get-hamming-weight [num]
;;             (if (< num 2)
;;               num
;;               (+ (get-hamming-weight (bit-shift-right num 1)) (bit-and num 1))))]
;;       (get-hamming-weight num)))
;; (map hamming-weight [2r1011 2r10000000 2r11111111111111111111111111111101])

;; ;;202
;; (defn is-happy [n]
;;   (letfn [(integer-digits [n]
;;             (if (< n 10)
;;               [n]
;;               (conj (integer-digits (quot n 10)) (rem n 10))))
;;           (square [n] (* n n))
;;           (is-happy' [results n]
;;            (let [result (apply + (map square (integer-digits n)))]
;;              (cond
;;                (= result 1) true
;;                (results result) false
;;                :else (is-happy' (conj results result) result))))]
;;     (is-happy' #{n} n)
;;     ))
;; (map is-happy [19 2])

;; ;;205
;; (defn is-isomorphic1 [s t]
;;   (let [->letter-map (fn [s1 s2]
;;                        (let [cs1 (vec s1)
;;                              cs2 (vec s2)
;;                              add (fn [m index]
;;                                    (assoc m (cs1 index) (cs2 index)))]
;;                          (reduce add {} (range (count s1)))))
;;         map1 (->letter-map s t)
;;         map2 (->letter-map t s)
;;         translate (fn [letter-map s]
;;                     (->> (reduce (fn [result c]
;;                                    (conj result (get letter-map c))) [] (vec s))
;;                          (str/join "")))]
;;     (and (= (translate map1 s) t) (= (translate map2 t) s))))
;; (defn is-isomorphic [s t] ;;not working
;;   (let [cs1 (vec s)
;;         cs2 (vec t)
;;         ascii (make-array Integer/TYPE 512)
;;         check (fn [result index]
;;                 (let [num1 (int (cs1 index))
;;                       num2 (+ (int (cs2 index)) 256)
;;                       value1 (aget ascii num1)
;;                       value2 (aget ascii num2)]
;;                   (if (not= value1 value2)
;;                     (reduced false)
;;                     (do
;;                       (aset ascii num1 (inc index))
;;                       (aset ascii num2 (inc index))
;;                       true))))]
;;     (reduce check true (range (count s)))))

;; (defn is-isomorphic3 [s t]
;;   (let [cs1 (vec s)
;;         cs2 (vec t)]
;;     (apply = (map count [(set (map vector cs1 cs2)) (set cs1) (set cs2)]))))
;; (map (partial apply is-isomorphic) ['("egg" "add") '("foo" "bar") '("paper" "title")])

;; ;;217
;; (defn contains-duplicate [nums]
;;   (let [add (fn [num-map num] (assoc num-map num 1))]
;;     (< (count (reduce add {} nums)) (count nums))))
;; (map contains-duplicate [[1 2 3 1] [1 2 3 4] [1 1 1 3 3 4 3 2 4 2]])

;; ;;219
;; (defn contains-nearby-duplicate [nums k]
;;   (let [len (count nums)
;;         abs (fn [n]
;;               (if (neg? n) (- 0 n) n))
;;         check-nearby-duplicate (fn [index]
;;                                  (reduce (fn [result index2]
;;                                            (let [num1 (nums index)
;;                                                  num2 (nums index2)]
;;                                              (if (and (= num1 num2) (<= (abs (- index index2)) k))
;;                                                (reduced true)
;;                                                false)))
;;                                          false (range (inc index) (+ index k 1))))]
;;     (reduce (fn [result index]
;;               (if (check-nearby-duplicate index)
;;                 (reduced true)
;;                 result))
;;             false (range (- len k)))))
;; (map (partial apply contains-nearby-duplicate) ['([1 2 3 1] 3) '([1 0 1 1] 1) '([1 2 3 1 2 3] 2)])

;; ;;228
;; (defn summary-ranges [nums]
;;   (let [len (count nums)
;;         ->range (fn [xs]
;;                   (if (= 1 (count xs))
;;                     (str (first xs))
;;                     (str (first xs) "->" (last xs))))
;;         add-range (fn [[results result] index]
;;                     (cond
;;                       (= index len) [(if (empty? result) results (conj results result)) []]
;;                       (empty? result) [results [(nums index)]]
;;                       (= (inc (last result)) (nums index)) [results (conj result (nums index))]
;;                       :else [(conj results result) [(nums index)]]))]
;;     (->> (reduce add-range [[] []] (range (inc len)))
;;          (first)
;;         (map ->range)
;;          )
;; ))
;; (map summary-ranges [[0 1 2 4 5 7] [0 2 3 4 6 8 9] [] [-1] [0]])

;; ;;231
;; (defn is-power-of-two [n]
;;   (letfn [(power-of-two? [n]
;;             (cond
;;               (= n 1) true
;;               (zero? (bit-and n 1)) (power-of-two? (bit-shift-right n 1))
;;               :else false))]
;;     (power-of-two? n)))
;; (map is-power-of-two [1 16 3 4 5])

;; ;;242
;; (defn is-anagram [s t]
;;   (= (frequencies s) (frequencies t)))
;; (map (partial apply is-anagram) [["anagram" "nagaram"] ["rat" "car"]])

;; ;; ;;243
;; (defn shortest-distance2 [words-dict word1 word2]
;;   (let [get-indices (fn [s] (->> (map-indexed vector words-dict)
;;                                  (filter (fn [[index word]]
;;                                            (= word s)))
;;                                     (map first)))
;;         indices1 (get-indices word1)
;;         indices2 (get-indices word2)
;;         index-pairs (for [index1 indices1 index2 indices2]
;;                       [index1 index2])
;;         abs (fn [n] (if (neg? n) (- 0 n) n))
;;         distance (fn [[index1 index2]] (abs (- index1 index2)))]
;;     (apply min (map distance index-pairs))))

;; (defn shortest-distance [words-dict word1 word2]
;;   (let [len (count words-dict)
;;         abs (fn [n] (if (neg? n) (- 0 n) n))
;;         get-min-distance' (fn [index1 index2 distance]
;;                             (if (not (or (nil? index1) (nil? index2)))
;;                               (min (abs (- index1 index2)) distance)
;;                               distance))
;;         get-min-distance (fn [[distance index1 index2] index]
;;                            (let [word (words-dict index)]
;;                              (cond
;;                                (= word word1) [(get-min-distance' index index2 distance) index index2]
;;                                (= word word2) [(get-min-distance' index1 index distance) index1 index]
;;                                :else [distance index1 index2])))]
;;     (first (reduce get-min-distance [len nil nil] (range len)))))
;; (map (partial apply shortest-distance) ['(["practice"  "makes"  "perfect"  "coding"  "makes"] "coding" "practice") '(["practice"  "makes"  "perfect"  "coding"  "makes"]  "makes"  "coding")])

;; ;;246
;; (defn is-strobogrammatic [num]
;;   (let [mirror-map {\6 \9 \9 \6 \8 \8 \1 \1 \0 \0}
;;         digits (vec num)
;;         len (count digits)
;;         compare-digit (fn [result index]
;;                         (let [digit1 (digits index)
;;                               digit2 (digits (- len 1 index))]
;;                           (= digit1 (get mirror-map digit2))))
;;         contains-non-mirror-digit? (fn [n]
;;                                      (not-empty (set/intersection (set (vec n)) #{\2 \3 \4 \5 \7})))
;;         strobogrammatic?' (fn [n]
;;                             (reduce compare-digit true (range (quot (count num) 2))))
;;         strobogrammatic? (fn [n]
;;                            (if (contains-non-mirror-digit? n)
;;                              false
;;                              (strobogrammatic?' n)))]
;;     (strobogrammatic? num)))
;; (map is-strobogrammatic ["69" "88" "962" "1"])

;; ;;252
;; (defn can-attend-meetings [intervals]
;;   (let [compare-interval (fn [[start1 end1] [start2 end2]]
;;                            (if (= start1 start2)
;;                              (compare end1 end2)
;;                              (compare start1 start2)))
;;         sorted-intervals (sort compare-interval intervals)
;;         busy? (fn [result index]
;;                 (let [[start1 end1] (nth sorted-intervals (dec index))
;;                       [start2 end2] (nth sorted-intervals index)]
;;                   (if (> end1 start2)
;;                     (reduced false)
;;                     result)))]
;;     (reduce busy? true (range 1 (count sorted-intervals)))))
;; (map can-attend-meetings [[[0 30] [5 10] [15 20]] [[7 10] [2 4]]])

;; ;;258
;; (defn add-digits [num]
;;   (letfn [(integer-digits [n]
;;             (if (< n 10)
;;               [n]
;;               (conj (integer-digits (quot n 10)) (rem n 10))))
;;           (sum [xs] (apply + xs))]
;;   (if (< num 10)
;;     num
;;     (add-digits (sum (integer-digits num))))))
;; (map add-digits [38 0])

;; ;; 263
;; (defn is-ugly [n]
;;   (letfn [(divisible? [a b]
;;             (zero? (rem a b)))
;;           (nested-divide [a b]
;;             (cond
;;               (not (divisible? a b)) a
;;               (and (divisible? a b) (= a b)) 1
;;               :else (nested-divide (quot a b) b)))
;;           (check [a b]
;;             (let [result (nested-divide a b)]
;;               (if (= 1 result)
;;                 (reduced 1)
;;                 result)))]
;;     (= 1 (reduce check n [2 3 5]))))

;; (map is-ugly [6 8 14 1])

;; ;;266
;; (defn can-permute-palindrome [s]
;;   (let [freqs (frequencies s)
;;         odds (filter odd? (vals freqs))]
;;    (= (count odds) 1)))
;; (map can-permute-palindrome ["code" "aab" "carerac"])

;; ;;268
;; (defn missing-number [nums]
;;   (let [n (count nums)
;;         num-set (set nums)]
;;     (reduce (fn [result num]
;;               (if (contains? num-set num)
;;                 result
;;                 (reduced num)))
;;             nil (range (inc n)))))
;; (map missing-number [[3 0 1] [0 1] [9 6 4 2 3 5 7 0 1] [0]])

;; ;;283
;; (defn move-zeroes [nums]
;;   (let [xs (into-array nums)
;;         move (fn [zeros index]
;;                (if (zero? (aget xs index))
;;                  (inc zeros)
;;                  (do
;;                    (aset xs (- index zeros) (aget xs index))
;;                    (aset xs index 0)
;;                    zeros)))]
;;     (reduce move 0 (range (count xs)))
;;     (vec xs)))
;; (map move-zeroes [[0 1 0 3 12] [0]])

;; ;;290
;; (defn word-pattern [pattern s]
;;   (let [cs (map str (vec pattern))
;;         words (re-seq #"\w+" s)
;;         pairs (map sort (concat (mapv vector cs words) (mapv vector words cs)))]
;;     (apply = (map (comp count set) [pairs cs words]))))
;; (map (partial apply word-pattern) [["abba" "dog cat cat dog"] ["abba" "dog cat cat fish"] ["aaaa" "dog cat cat dog"] ["abba" "dog dog dog dog"]])

;; ;;292
;;  (defn can-win-nim [n]
;;    (not= 0 (rem n 4)))
;;  (map can-win-nim [4 1 2])

;; ;; ;;293
;; (defn generate-possible-next-moves [current-state]
;;   (let [len (count current-state)
;;         generate' (fn [index]
;;                     (str (subs current-state 0 index)
;;                          "--"
;;                          (subs current-state (+ index 2))))
;;         generate (fn [results index]
;;                    (let [s (subs current-state index (+ index 2))]
;;                      (if (= s "++")
;;                        (conj results (generate' index))
;;                        results)))]
;;     (reduce generate [] (range (dec len)))))
;; (map generate-possible-next-moves ["++++" "+"])

;; ;;326
;; (defn is-power-of-three [n]
;;   (let [abs (fn [n] (if (neg? n) (- 0 n) n))]
;;     (cond
;;       (= n 0) false
;;       (= (abs n) 1) true
;;       (zero? (rem n 3)) (is-power-of-three (quot n 3))
;;       :else false
;;   )))
;; (map is-power-of-three [27 0 9 45])

;; ;;338
;; (defn count-bits [n]
;;   (letfn [(count-bits' [n]
;;             (if (< n 2)
;;               n
;;               (+ (count-bits' (bit-shift-right n 1))
;;                  (bit-and n 1))))]
;;     (map count-bits' (range (inc n)))))
;; (map count-bits [2 5])

;; ;;342
;; (defn is-power-of-four [n]
;;   (letfn [(abs [n] (if (neg? n) (- 0 n) n))
;;           (is-power-of-four' [n]
;;             (cond
;;               (zero? n) false
;;               (= 1 (abs n)) true
;;               (zero? (rem n 4)) (is-power-of-four' (quot n 4))
;;               :else false))]
;;     (is-power-of-four' n)))
;; (map is-power-of-four [16 5 1])

;; ;; 1248
;; ;; (defn number-of-subarrays [nums k]
;; ;;   (let [indices (vec (filter (fn [index] (odd? (nums index))) (range (count nums))))
;; ;;         len (count indices)
;; ;;         ]
;; ;;     (map (fn [i]
;; ;;            (let [start (indices i)]
;; ;;                  (if (< (+ start k) len)
;; ;;                        (- (indices (+ start k)) (indices (+ start (dec k))))
;; ;;                        1))) (range (- len (dec k))))))

;; (defn number-of-subarrays1 [nums k]
;;   (let [dp (make-array Long/TYPE 10)
;;     n (reduce (fn [index num]
;;               (if (odd? num)
;;                 (do
;;                 (aset dp (inc index) 0)
;;                 (inc index))
;;                 (do
;;                   (aset dp index (inc (aget dp index)))
;;                   index
;;                 )
;;               ))
;;             0 nums
;;             )]
;;     (reduce (fn [result i] (+ result (* (inc (aget dp (- i k))) (inc (aget dp i)))))
;;             0 (range k (inc n)))
;;     ))

;; ;; (defn number-of-subarrays [nums k]
;; ;;   (let [1000])
;; ;;   )

;; ;(map (partial apply number-of-subarrays) ['([1 1 2 1 1] 3) '([2 4 6] 1) '([2 2 2 1 2 2 1 2 2 2] 2)])
;; ;;344
;; (defn reverse-string [s]
;;   (let [len (count s)
;;         cs (into-array (vec s))]
;;     (doseq [index (range (quot len 2))]
;;       (let [left index
;;             right (- (dec len) left)
;;             c (aget cs left)]
;;       (aset cs left (aget cs right))
;;       (aset cs right c)))
;;     (vec cs)))
;; (map reverse-string ["hello" "Hannah"])

;; ;; ;;345
;; (defn reverse-vowels1 [s]
;;   (let [len (count s)
;;         cs (vec s)
;;         vowels #{\a \e \i \o \u \A \E \I \O \U}
;;         vowel? #(contains? vowels %)
;;         indices (vec (filter #(vowel? (cs %)) (range len)))
;;         result (into-array cs)
;;         indices-len (count indices)]
;;     (doseq [index (range (quot indices-len 2))]
;;       (let [left index
;;             right (- (dec indices-len) left)
;;             c (aget result (indices left))]
;;         (aset result (indices left) (aget result (indices right)))
;;         (aset result (indices right) c)))
;;       (vec result)))

;; (defn reverse-vowels [s]
;;   (let [cs (re-seq #"[aeiouAEIOU]" s)] 
;;     (apply
;;      (partial format (str/replace s #"[aeiouAEIOU]" "%s")) (reverse cs))))
;; (map reverse-vowels ["hello" "leetcode"])

;; ;;349
;; (defn intersection [nums1 nums2]
;;   (vec (set/intersection (set nums1) (set nums2)))
;;   )
;; (map (partial apply intersection) ['([1 2 2 1] [2 2]) '([4 9 5] [9 4 9 8 4])])

;; ;;350
;; (defn intersect [nums1 nums2]
;;   (let [freqs1 (frequencies nums1)
;;         freqs2 (frequencies nums2)
;;         append (fn [result num]
;;                  (let [cnt1 (get freqs1 num)
;;                        cnt2 (get freqs2 num)]
;;                    (if (not (nil? cnt2))
;;                      (concat result (mapv (fn [_] num) (range (min cnt1 cnt2))))
;;                      result
;;                      )
;;                    ))]
;;     (reduce append [] (keys freqs1))))
;; (map (partial apply intersect) ['([1 2 2 1] [2 2]) '([4 9 5] [9 4 9 8 4])])

;; ;;367
;; (defn is-perfect-square [num]
;;   (letfn [(square [n] (* n n))
;;           (perfect-square? [n start end]
;;             (let [middle (quot (+ start end) 2)
;;                   product (square middle)]
;;               (cond
;;                 (> start end) false
;;                 (> product n) (perfect-square? n start (dec middle))
;;                 (< product n) (perfect-square? n (inc middle) end)
;;                 :else true)))]
;;     (perfect-square? num 1 num)))
;; (map is-perfect-square [16 14])

;; ;;383
;; (defn can-construct [ransom-note magazine]
;;   (let [freqs1 (frequencies ransom-note)
;;         freqs2 (frequencies magazine)
;;         check (fn [result letter]
;;                 (let [cnt1 (get freqs1 letter)
;;                       cnt2 (or (get freqs2 letter) 0)]
;;                   (if (< cnt2 cnt1)
;;                     (reduced false)
;;                     result)))]
;;     (reduce check true (keys freqs1))))
;; (map (partial apply can-construct) ['("a" "b") '("aa" "ab") '("aa" "aab")])

;; ;;387
;; (defn first-uniq-char [s]
;;   (let [cs (vec s)
;;         freqs (frequencies cs)
;;     letter (reduce (fn [result letter]
;;               (if (= (get freqs letter) 1)
;;                 (reduced letter)
;;                 result)) nil (keys freqs))]
;;     (.indexOf cs letter)
;; ))
;; (map first-uniq-char ["leetcode" "loveleetcode" "aabb"])

;; ;;389
;; (defn find-difference [s t]
;;   (let [freqs1 (frequencies (vec s))
;;         freqs2 (frequencies (vec t))
;;         check (fn [result c]
;;                 (let [cnt1 (get freqs1 c)
;;                       cnt2 (get freqs2 c)]
;;                   (if (or (nil? cnt1) (= (inc cnt1) cnt2))
;;                     c
;;                     result)))]
;;     (reduce check nil (keys freqs2))))
;; (map (partial apply find-difference) ['("abcd" "abcde") '("" "y") '("a" "aa") '("ae" "aea")])

;; ;;392
;; (defn is-subsequence [s t]
;;   (let [cs1 (vec s)
;;         cs2 (vec t)
;;         rows (count s)
;;         cols (count t)
;;         dp (make-array Long/TYPE (inc rows) (inc cols))]
;;     (doseq [r (range 1 (inc rows)) c (range 1 (inc cols))]
;;       (let [max-score' (max (aget dp (dec r) (dec c)) (aget dp r (dec c)))
;;             score (if (= (cs1 (dec r)) (cs2 (dec c))) 1 0)
;;             max-score (+ max-score' score)]
;;         (aset dp r c max-score)))
;;     (= rows (aget dp rows cols))))
;; (map (partial apply is-subsequence) ['("abc" "ahbgdc") '("axc" "ahbgdc")])

;; ;;401
;; (defn read-binary-watch [turned-on]
;;   (letfn [(integer-digits' [n]
;;             (if (< n 2)
;;               [n]
;;               (conj (integer-digits' (quot n 2)) (rem n 2))))]
;;     (let [->time (fn [h m]
;;                    (let [padding (fn [n] (if (< n 10) (str "0" n) (str n)))]
;;                      (str h ":" (padding m))))
;;           integer-digits (fn [n]
;;                            (if (zero? n)
;;                              [0]
;;                              (integer-digits' n)))

;;           sum-digits (fn [n]
;;                        (apply + (integer-digits n)))
;;           watch-times (for [h (range 12) m (range 60)]
;;                         [(->time h m) (+ (sum-digits h) (sum-digits m))])]
;;       (->> watch-times
;;            (filter #(= (last %) turned-on))
;;            (map first)))))
;; (map read-binary-watch [1 9])

;; ;;405
;; (defn to-hex [num]
;;   (let [hex-map {0 \0 1 \1 2 \2 3 \3 4 \4 5 \5 6 \6 7 \7 8 \8 9 \9 10 \a 11 \b 12 \c 13 \d 14 \e 15 \f}]
;;     (letfn [(->hex-digits [n]
;;               (if (< n 16)
;;                 [(get hex-map n)]
;;                 (conj (->hex-digits (quot n 16)) (get hex-map (rem n 16)))))
;;             (->hex [n]
;;               (str/join "" (->hex-digits n)))]
;;       (cond
;;         (pos? num) (->hex num)
;;         (neg? num) (->hex (+ (bit-shift-left 1 32) num))
;;         :else "0"))))
;; (map to-hex [26 -1])

;; ;;408
;; (defn valid-word-abbreviation [word abbr]
;;   (let [lens  (re-seq #"\d+" abbr)
;;         tokens (map (fn [len] (apply str (repeat (Integer/parseInt len) "0"))) lens)
;;         leading-zero? (fn []
;;                         (->> lens
;;                              (filter (fn [digits] (str/starts-with? digits "0")))
;;                              ((comp not empty?))))
;;         expand-abbr (fn [abbr]
;;                       (apply (partial format (str/replace abbr #"\d+" "%s")) tokens))
;;         valid-word-abbr? (fn [word abbr]
;;                            (let [cs1 (vec word)
;;                                  cs2 (vec (expand-abbr abbr))
;;                                  compare-char (fn [result index]
;;                                                 (if (or (= (cs1 index) (cs2 index)) (= \0 (cs2 index)))
;;                                                   result
;;                                                   (reduced false)))]
;;                              (if (not= (count cs1) (count cs2))
;;                                false
;;                                (reduce compare-char true (range (count cs1))))))]
;;     (if (leading-zero?)
;;       false
;;       (valid-word-abbr? word abbr))))
;; (map (partial apply valid-word-abbreviation) ['("internationalization" "i12iz4n") '("apple" "a2e")])
;; ;;409
;; (defn longest-palindrome [s]
;;   (let [counts (vals (frequencies (vec s)))
;;         odds (filter odd? counts)
;;         evens (filter even? counts)
;;         max-odd-number (if (empty? odds) 0 (apply max odds))]
;;     (apply + (conj evens max-odd-number))))
;; (map longest-palindrome ["abccccdd" "a" "bb"])

;; ;;412
;; (defn fizz-buzz [n]
;;   (let [divisible? (fn [a b]
;;                      (zero? (rem a b)))
;;         divisible-by-3 #(divisible? % 3)
;;         divisible-by-5 #(divisible? % 5)
;;         ->fizz-buzz (fn [n]
;;                       (cond
;;                         (and (divisible-by-3 n) (divisible-by-5 n)) "FizzBuzz"
;;                         (divisible-by-3 n) "Fizz"
;;                         (divisible-by-5 n) "Buzz"
;;                         :else (str n)))]
;;     (map ->fizz-buzz (range 1 (inc n)))))
;; (map fizz-buzz [3 5 15])

;; ;;414
;; (defn third-max [nums]
;;   (let [max3 (fn [[m1 m2 m3] num]
;;                (cond
;;                  (< m1 num) [num m1 m2]
;;                  (and (> m1 num) (nil? m2)) [m1 num m3]
;;                  (and (> m1 num) (> num m2)) [m1 num m2]
;;                  (and (> m1 num) (< num m2) (or (nil? m3) (> num m3))) [m1 m2 num]
;;                  :else [m1 m2 m3]))
;;         results (reduce max3 [(first nums) nil nil] (rest nums))]
;;     (if (nil? (last results))
;;       (first results)
;;       (last results))))
;; (map third-max [[3 2 1] [1 2] [2 2 3 1]])

;; ;;415
;; (defn add-strings [num1 num2]
;;   (let [to-digit (fn [c] (- (int c) (int \0)))
;;         integer-digits (fn [s]
;;                        (->> (map to-digit (vec s))
;;                             reverse
;;                             vec))
;;         digits1 (integer-digits num1)
;;         digits2 (integer-digits num2)
;;         len1 (count digits1)
;;         len2 (count digits2)
;;         add (fn [[results carry] index]
;;               (let [digit1 (if (< index len1)
;;                              (digits1 index)
;;                              0)
;;                     digit2 (if (< index len2)
;;                              (digits2 index)
;;                              0)
;;                     sum (+ digit1 digit2 carry)]

;;                 [(cons (rem sum 10) results) (quot sum 10)]))
;;         to-char (fn [digit] (char (+ digit (int \0))))
;;         [results carry] (reduce add [[] 0] (range (max len1 len2)))
;;         digits (if (zero? carry) results (cons carry results))]
;;     (->> (map to-char digits)
;;          (str/join ""))))
;; (map (partial apply add-strings) ['("11" "123") '("456" "77") '("0" "0")])

;; ;;422
;; (defn valid-word-square [words]
;;   (let [pad-right (fn [len s]
;;                     (if (= (count s) len)
;;                       s
;;                       (str s (str/join "" (repeat (- len (count s)) " ")))))
;;         transpose (fn [m]
;;                     (let [rows (count m)
;;                           cols (count (m 0))
;;                           indices (for [r (range rows) c (range cols)]
;;                                     [r c])
;;                           mt (make-array Character/TYPE cols rows)]
;;                       (doseq [[r c] indices]
;;                         (aset mt c r ((m r) c)))
;;                       (mapv vec mt)))
;;         len (count words)
;;         matrix (mapv vec (map #(pad-right len %) words))]
;;     (= matrix (transpose matrix))))
;; (map valid-word-square [["abcd" "bnrt" "crmy" "dtye"] ["abcd" "bnrt" "crm" "dt"] ["ball" "area" "read" "lady"]])

;; ;;434
;; (defn count-segments [s]
;;   ;(count (re-seq #"[^ ]+" s))
;;   (count (filter #(pos? (count %)) (str/split s #"\s+"))))
;; (map count-segments ["Hello, my name is John" "Hello" "love live! mu'sic forever" ""])

;; ;;441
;; (defn arrange-coins [n]
;;   (let [check-complete-rows (fn [n index]
;;                               (let [sum (quot (* index (inc index)) 2)
;;                                     remained (- n sum)]
;;                                 (if (< remained (inc index))
;;                                   (reduced index)
;;                                   remained)))]
;;   (reduce check-complete-rows n (range 1 (inc n)))))
;; (map arrange-coins [5 8])

;; ;;448
;; (defn find-disappeared-numbers1 [nums]
;;   (vec (set/difference (set (range 1 (inc (count nums)))) (set nums))))
;; (defn find-disappeared-numbers [nums]
;;   (let [xs (into-array nums)
;;         abs (fn [n] (if (neg? n) (- 0 n) n))]
;;     (doseq [index (range (count nums))]
;;       (let [num (dec (abs (aget xs index)))]
;;         (if (pos? (aget xs num))
;;           (aset xs num (* -1 (aget xs num))))))
;;     (filter #(pos? (aget xs (dec %))) (range 1 (inc (count nums))))))
;; (map find-disappeared-numbers [[4 3 2 7 8 2 3 1] [1 1]])

;; ;;455
;; (defn find-content-children [g s]
;;   (let [greeds (vec (sort g))
;;         sizes (vec (sort s))
;;         add-content-child (fn [[result index] size]
;;                               (cond
;;                                 (= index (count greeds)) [result index]
;;                                 (>= size (greeds index)) [(inc result) (inc index)]
;;                                 :else [result index]))]
;;    (first (reduce add-content-child [0 0] sizes))))
;; (map (partial apply find-content-children) [[[1 2 3] [1 1]] [[1 2] [1 2 3]]])

;; ;;459
;; (defn repeated-substring-pattern1 [s]
;;   (letfn [(gcd [a b]
;;             (if (zero? (rem a b))
;;               b
;;               (gcd b (quot a b))))]
;;     (let [sizes (vals (frequencies (vec s)))
;;           len (count s)
;;           max-parts (reduce (fn [result size]
;;                               (if (> result size)
;;                                 (gcd result size)
;;                                 (gcd size result))) (first sizes) (rest sizes))
;;           compare-substrings (fn [result parts]
;;                                (let [size (quot len parts)
;;                                      compare-strings (fn [result index]
;;                                                        (let [start (* index size)]
;;                                                          (if (= (subs s 0 size)
;;                                                                 (subs s start (+ start size)))
;;                                                            result
;;                                                            (reduced false))))]
;;                                  (if (zero? (rem len parts))
;;                                    (if (reduce compare-strings true (range 1 parts))
;;                                      (reduced true)
;;                                      false)
;;                                  result)))]

;;       (reduce compare-substrings false (range 2 (inc max-parts))))))

;; (defn repeated-substring-pattern [s]
;;   (let [ss (str s s)
;;         len (count ss)]
;;     (not= -1 (.indexOf (subs ss 1 (dec len)) s))))
;; (map repeated-substring-pattern ["abab" "aba" "abcabcabcabc"])

;; ;;461
;; (defn hamming-distance [x y]
;;   (letfn [(distance [x y]
;;             (let [xor (fn [x y]
;;                         (bit-xor (bit-and x 1) (bit-and y 1)))
;;                   x' (bit-shift-right x 1)
;;                   y' (bit-shift-right y 1)]
;;               (if (and (zero? x) (zero? y))
;;                 0
;;                 (+ (xor x y) (distance x' y')))))]
;;     (distance x y)))
;; (map (partial apply hamming-distance) [[1 4] [3 1] [3 11]])

;; ;;463
;; (defn island-perimeter [grid]
;;   (let [row (count grid)
;;         col (count (grid 0))
;;         indices (for [r (range row) c (range col)]
;;                   [r c])
;;         not-land-cell? (fn [[r c]]
;;                          (or (neg? r) (neg? c) (>= r row) (>= c col)
;;                              (zero? ((grid r) c))))
;;         land-cell? (fn [[r c]]
;;                      (= 1 ((grid r) c)))
;;         cell-perimeter (fn [[r c]]
;;                          (let [neighbours [[(inc r) c] [(dec r) c] [r (inc c)] [r (dec c)]]]
;;                               (count (filter true? (map not-land-cell? neighbours)))))]
;;     (->> (filter land-cell? indices)
;;          (map cell-perimeter)
;;          (apply +))))
;; (map island-perimeter [[[0 1 0 0] [1 1 1 0] [0 1 0 0] [1 1 0 0]] [[1]] [[1 0]]])

;; ;;476
;; (defn find-complement [num]
;;   (letfn [(complement-integer-digits [n]
;;             (if (< n 2)
;;               [(bit-xor n 1)]
;;               (conj (complement-integer-digits (bit-shift-right n 1))
;;                     (bit-xor (bit-and n 1) 1))))]
;;     (let [digits (complement-integer-digits num)]
;;       (reduce #(+ (bit-shift-left %1 1) %2) 0 digits))))
;; (map find-complement [5 1])

;; ;;482
;; (defn license-key-formatting [s k]
;;   (let [words (str/split s #"-")
;;         rest-string (str/join "" (rest words))
;;         new-words (cons (first words) (map (partial str/join "") (partition k (vec rest-string))))]
;;     (str/join "-" new-words)))
;;  (map (partial apply license-key-formatting) [["5F3Z-2e-9-w" 4] ["2-5g-3-J" 2]])

;; ;;485
;; (defn find-max-consecutive-ones [nums]
;;   (let [find-max-ones (fn [[max-ones ones] num]
;;                         (if (= num 1)
;;                           [(max max-ones (inc ones)) (inc ones)]
;;                           [max-ones 0]))
;;         initial [(first nums) (first nums)]]
;;     (first (reduce find-max-ones initial (rest nums)))))
;; (map find-max-consecutive-ones [[1 1 0 1 1 1] [1 0 1 1 0 1]])

;; ;;492
;; (defn construct-rectangle [area]
;;   (let [widths (range 2 (inc (Math/sqrt area)))
;;         find-rectangle (fn [[delta [height width]] w]
;;                          (let [h (quot area w)]
;;                          (if (and (zero? (rem area w)) (> delta (- h w)))
;;                              [(- h w) [(quot area w) w]]
;;                              [delta [height width]])))]
;;     (->> (reduce find-rectangle [(- area 1) [area 1]] widths)
;;          (last)
;;          )))
;; (map construct-rectangle [4 37 122122])

;; ;;495
;; (defn find-poisoned-duration [time-series duration]
;;   (let [count-duration (fn [result index]
;;                          (let [t0 (time-series (dec index))
;;                                t1 (time-series index)
;;                                interval (- t1 t0)]
;;                            (+ result (min duration interval))))
;;         time-range (range 1 (count time-series))]
;;     (reduce count-duration 2 time-range)))
;; (map (partial apply find-poisoned-duration) [[[1 4] 2] [[1 2] 2]])

;; ;;496
;; (defn next-greater-element [nums1 nums2]
;;   (let [indexed-nums (map vector (range (count nums2)) nums2)
;;         add-num-index (fn [result [index num]]
;;                         (let [indices (or (get result num) [])]
;;                           (assoc result num (conj indices index))))
;;         indices-map (reduce add-num-index {} indexed-nums)
;;         next-greater (fn [num]
;;                        (let [index (.indexOf nums2 num)
;;                              elements (range (inc num) 1001)
;;                              result (reduce (fn [result n]
;;                                               (let [indices (get indices-map n)
;;                                                     greater-element-index (first (filter #(> % index) indices))]
;;                                                 (if (or (empty? indices) (nil? greater-element-index))
;;                                                   result
;;                                                   (reduced n))))
;;                                             nil elements)]
;;                          (or result -1)))]
;;     (mapv next-greater nums1)))
;; (map (partial apply next-greater-element) [[[4 1 2] [1 3 4 2]] [[2 4] [1 2 3 4]]])

;; ;;500
;; (defn find-words [words]
;;   (let [row1 (set (vec "qwertyuiop"))
;;         row2 (set (vec "asdfghjkl"))
;;         row3 (set (vec "zxcvbnm"))
;;         in-one-row? (fn [word]
;;                       (let [letter-set (set (vec (str/lower-case word)))
;;                             rows [row1 row2 row3]
;;                             subset? #(set/subset? letter-set %)]
;;                         (->> (map subset? rows)
;;                              (filter true?)
;;                              (count)
;;                              (pos?))))]
;;     (filter in-one-row? words)))
;; (map find-words [["Hello" "Alaska" "Dad" "Peace"] ["omk"] ["adsdf" "sfd"]])

;; ;;504
;; (defn convert-to-base7 [num]
;;   (let [abs (fn [n] (if (neg? n) (- 0 n) n))]
;;     (letfn [(convert' [n]
;;               (if (< n 7)
;;                 (str n)
;;                 (str (convert' (quot n 7)) (rem n 7))))
;;             (convert [n]
;;               (cond
;;                 (pos? n) (convert' n)
;;                 (neg? n) (str "-" (convert' (abs n)))
;;                 :else "0"))]
;;       (convert num))))
;; (map convert-to-base7 [100 -7])

;; ;;506
;; (defn find-relative-ranks [score]
;;   (let [score-map (->> (map vector (vec (sort > score)) (range 1 (inc (count score))))
;;                        (reduce (fn [result [s index]] (assoc result s index)) {}))
;;         ->rank (fn [s]
;;                  (let [index (get score-map s)]
;;                    (case index
;;                      1 "Gold Medal"
;;                      2 "Silver Medal"
;;                      3 "Bronze Medal"
;;                      (str index))))]
;;     (map ->rank score)))
;; (map find-relative-ranks [[5 4 3 2 1] [10 3 8 9 4]])

;; ;;507
;; (defn check-perfect-number [num]
;;   (let [factors (distinct (reduce (fn [result m]
;;                                   (if (zero? (rem num m))
;;                                     (conj result m (quot num m))
;;                                     result)) [1] (range 2 (inc (Math/sqrt num)))))]
;;     (= (apply + factors) num)))
;; (map check-perfect-number [28 6 496 8128 2])

;; ;;509
;; (defn fib [n]
;;   (letfn [(fib' [a b]
;;             [b (+ a b)])]
;;    (first (reduce (fn [[a b] _]
;;               (fib' a b)) [0 1] (range n)))))
;; (map fib [2 3 4 5])

;; ;;520
;; (defn detect-capital-user [word]
;;   (or (= word (str/upper-case word))
;;       (= word (str/lower-case word))
;;       (= word (str/capitalize word))))
;; (map detect-capital-user ["USA" "FlaG"])

;; ;;521
;; ;; (defn find-LUS-length [a b]
;; ;;   (let [pattern (re-pattern (format "[%s]+" (str/join "" (set (vec b)))))
;; ;;         subsequences (str/split a pattern)
;; ;;         max-subsequence-length #(apply max (map count subsequences))]
;; ;;     (if (= a b)
;; ;;       -1
;; ;;       (max-subsequence-length))))
;; (defn find-LUS-length [a b]
;;   (if (= a b)
;;     -1
;;     (max (count a) (count b))))
;; (map (partial apply find-LUS-length) [["aba" "cdc"] ["aaa" "bbb"] ["aaa" "aaa"]])

;; ;;541
;; (defn reverse-str [s k]
;;   (let [parts (vec (partition k (vec s)))
;;         append-str (fn [result index]
;;                  (let [part (if (even? index) (reverse (parts index)) (parts index))]
;;                    (conj result part)))]
;;     (->>  (reduce append-str [] (range (count parts)))
;;           (flatten)
;;           (str/join ""))))
;; (map (partial apply reverse-str) [["abcdefg" 2] ["abcd" 2]])

;; ;; ;;551
;; (defn check-record [s]
;;   (let [freqs (frequencies (vec s))
;;         absents (or (freqs \A) 0)
;;         lates (map count (re-seq #"L+" s))
;;         max-lates (if (empty? lates)
;;                     0
;;                     (apply max lates))]
;;     (and (< absents 2) (< max-lates 3))))
;; (map check-record ["PPALLP" "PPALLL"])

;; ;;557
;; (defn reverse-words [s]
;;   (let [words (str/split s #"\s")]
;;     (str/join " " (map str/reverse words))))
;; (map reverse-words ["Let's take LeetCode contest" "God Ding"])

;; ;;561
;; (defn array-pair-sum [nums]
;;   (let [xs (vec (sort nums))
;;         len (count nums)
;;         add (fn [result index] (+ result (xs index)))]
;;     (reduce add 0 (range 0 len 2))))
;; (map array-pair-sum [[1 4 3 2] [6 2 6 5 1 2]])

;; ;;566
;; (defn matrix-reshape [mat r c]
;;   (let [row (count mat)
;;         col (count (mat 0))
;;         reshape (fn [m r c]
;;                   (let [matrix (make-array Long/TYPE r c)]
;;                     (doseq [i (range r) j (range c)]
;;                       (let [index (+ (* i c) j)
;;                             i' (quot index col)
;;                             j' (rem index col)]
;;                         (aset matrix i j ((m i') j'))))
;;                     (mapv vec matrix)))]
;;     (if (not= (* r c) (* row col))
;;       mat
;;       (reshape mat r c))))
;; (map (partial apply matrix-reshape) [[[[1 2] [3 4]] 1 4] [[[1 2] [3 4]] 2 4]])

;; ;;575
;; (defn distribute-candies [candy-type]
;;   (let [freqs (frequencies candy-type)
;;         types (count (keys freqs))
;;         len (quot (count candy-type) 2)]
;;     (min types len)
;;     )
;;   )
;; (map distribute-candies [[1 1 2 2 3 3] [1 1 2 3] [6 6 6 6]])

;; ;;594
;; ;; (defn find-LHS [nums]
;; ;;   (let [add-index (fn [m [index num]]
;; ;;                     (let [indices (or (get m num) [])]
;; ;;                       (assoc m num (conj indices index))))
;; ;;         index-map (reduce add-index {} (map vector (range (count nums)) nums))
;; ;;         max-length (fn [num]
;; ;;                      (let [indices1 (get index-map num)
;; ;;                            indices2 (get index-map (inc num))]
;; ;;                      (if (empty? indices2)
;; ;;                        0
;; ;;                        (+ (count indices1) (count indices2)))))
;; ;;         ]
;; ;;     (apply max (map max-length (sort (keys index-map))))))
;; ;; (map find-LHS [[1 3 2 2 5 2 3 7] [1 2 3 4] [1 1 1 1]])

;; ;;598
;; (defn max-count [m n ops]
;;   (let [matrix (make-array Long/TYPE m n)
;;         operation-counts (into [] (frequencies ops))]
;;     (doseq [[[row col] cnt] operation-counts]
;;       (doseq [r (range row) c (range col)]
;;         (aset matrix r c (+ (aget matrix r c) cnt))))
;;     (let [max-value (apply max (mapv #(apply max %) matrix))
;;           count-max #(count (filter (fn [x] (= max-value x)) (vec %)))]
;;       (->> (map count-max matrix)
;;            (apply +)))))
;; (map (partial apply max-count) ['(3 3 [[2 2] [3 3]]) '(3 3 [])])

;; ;;24
;; ;; (defn judge-point-24 [cards]
;; ;;   (letfn [(permutations [nums]
;; ;;             (vec (set (if (= 1 (count nums))
;; ;;                         nums
;; ;;                         (for [[index num] (map vector (range (count nums)) nums)
;; ;;                               rest-nums (permutations (vec (concat (subvec nums 0 index) (subvec nums (inc index)))))]
;; ;;                           (vec (flatten [num rest-nums])))))))]
;; ;;     (let [operators [(fn [a b] (+ a b))
;; ;;                      (fn [a b] (- a b))
;; ;;                      (fn [a b] (* a b))
;; ;;                      (fn [a b] (if (zero? b) 1/9999 (/ a b)))]

;; ;;           operators-list (for [op0 operators op1 operators op2 operators]
;; ;;                            [op0 op1 op2])
;; ;;           point-24? (fn [[[c0 c1 c2 c3] [op0 op1 op2]]]
;; ;;                       (true? (some #(= 24 %) [(op2
;; ;;                                                (op1
;; ;;                                                 (op0 c0 c1) c2) c3)

;; ;;                                               (op2
;; ;;                                                (op1 c0
;; ;;                                                     (op0 c1 c2)) c3)
;; ;;                                               (op2
;; ;;                                                (op0 c0 c1)
;; ;;                                                (op1 c2 c3))

;; ;;                                               (op2
;; ;;                                                (op1 c0
;; ;;                                                     (op0 c1 c2)) c3)

;; ;;                                               (op2 c0
;; ;;                                                    (op1 c1
;; ;;                                                         (op0 c2 c3)))])))
;; ;;           cards-list (permutations cards)
;; ;;           any-true? #(true? (some true? %))]
;; ;;       (->> (for [cards cards-list ops operators-list] [cards ops])
;; ;;            (map point-24?)
;; ;;            (any-true?)))))
;; ;; (map judge-point-24 [[4 1 8 7] [1 2 1 2]])

;; ;;605
;; (defn can-place-flowers [flowerbed n]
;;   (let [bed (vec (concat [0] (conj flowerbed 0)))
;;         plant (fn [start] (reduce (fn [result index]
;;               (cond
;;                 (zero? result) (reduced true)
;;                 (every? zero? [(bed (dec index)) (bed index) (bed (inc index))]) (if (zero? (dec result))
;;                                                                                    (reduced 0)
;;                                                                                    (dec result))
;;                 :else result))
;;                                       n (range start (dec (count bed)))))]
;;     (zero? (plant 1))))
;; (map (partial apply can-place-flowers) ['([1 0 0 0 1] 1) '([1 0 0 0 1] 2)])

;; ;;628
;; (defn maximum-product [nums]
;;   (let [nums (vec (sort < nums))
;;         len (count nums)]
;;    (max (* (nums (- len 1)) (nums (- len 2)) (nums (- len 3)))
;;     (* (nums 0) (nums 1) (nums (- len 1))))))
;; (map maximum-product ['(1 2 3) '(1 2 3 4) '(-1 -2 -3)])

;; ;;643
;; (defn find-max-average [nums k]
;;   (let [len (count nums)
;;         initial (apply + (take k nums))
;;         find-max-sum (fn [[max-sum sum] index]
;;                        (let [new-sum (- (+ sum (nums index)) (nums (- index k)))]
;;                          [(max max-sum new-sum) new-sum]))]
;;     (->> (reduce find-max-sum [initial initial] (range k len))
;;          (first)
;;          (#(float (/ % k))))))
;; (map (partial apply find-max-average) ['([1 12 -5 -6 50 3] 4) '([5] 1)])

;; ;; ;;645
;; (defn find-error-nums [nums]
;;   (let [len (count nums)
;;         freqs (frequencies nums)
;;         find-missing-or-repetitive (fn [result num]
;;                      (let [cnt (get freqs num)]
;;                        (if (or (nil? cnt) (> cnt 1))
;;                          (conj result num)
;;                          result)))]
;;     (reduce find-missing-or-repetitive [] (range 1 (inc len)))))
;; (map find-error-nums [[1 2 2 4] [1 1]])

;; ;;657
;; (defn judge-circle [moves]
;;   (let [freqs (frequencies (vec moves))]
;;     (and (= (get freqs \L) (get freqs \R))
;;          (= (get freqs \U) (get freqs \D)))))
;; (map judge-circle ["UD" "LL" "RRDD" "LDRRLRUULR"])

;; ;;661
;; (defn image-smoother [img]
;;   (let [row (count img)
;;         col (count (img 0))
;;         matrix (make-array Long/TYPE row col)
;;         indices (for [r (range row) c (range col)]
;;                   [r c])
;;         smooth (fn [r c]
;;                  (let [cell-values (for [i (range (max (dec r) 0) (min (+ 2 r) row))
;;                                          j (range (max (dec c) 0) (min (+ 2 c) col))]
;;                                      ((img i) j))]
;;                    (quot (apply + cell-values)
;;                          (count cell-values))))]
;;     (doseq [[r c] indices]
;;       (aset matrix r c (smooth r c)))
;;     ;(smooth row col 0 0)
;;     (mapv vec matrix)))
;; (map image-smoother [[[1 1 1] [1 0 1] [1 1 1]] [[100 200 100] [200 50 200] [100 200 100]]])

;; ;;674
;; (defn find-length-of-LCIS [nums]
;;   (let [compare-length (fn [[max-length length] index]
;;                          (if (> (nums index) (nums (dec index)))
;;                            [(max max-length (inc length)) (inc length)]
;;                            [max-length 1]))]
;;     (first (reduce compare-length [1 1] (range 1 (count nums))))))
;; (mapv find-length-of-LCIS [[1 3 5 4 7] [2 2 2 2 2]])

;; ;;680
;; (defn valid-palindrome1 [s]
;;   (let [len (count s)
;;         cs (vec s)
;;         compare-chars (fn [index] (= (cs index) (cs (- len 1 index))))
;;         results (map compare-chars (range len))
;;         parlindrome? #(or (and (odd? len) (zero? %))
;;                          (and (even? len) (< (quot % 2) 2)))]
;;     (->> results
;;          (filter false?)
;;          (count)
;;          (parlindrome?))))

;; (defn valid-palindrome [s]
;;   (let [len (count s)
;;         cs (vec s)]
;;     (letfn [(palindrome? [cs left right deleted]
;;               (cond
;;                 (= left right) true
;;                 (and (not= (cs left) (cs right)) (not deleted)) (palindrome? cs (inc left) right true)
;;                 (and (not= (cs left) (cs right)) deleted) false
;;                 :else (palindrome? cs (inc left) (dec right) deleted))
;;               )]
;;       (palindrome? cs 0 (dec len) false)
;;     )))
;; (map valid-palindrome ["aba" "abca" "abc"])

;; ;;682
;; (defn cal-points [ops]
;;   (let [keep-score (fn [scores op]
;;                      (case op
;;                       "+" (cons (+ (first scores) (second scores)) scores)
;;                       "D" (cons (* (first scores) 2) scores)
;;                       "C" (rest scores)
;;                       (cons (Integer/parseInt op) scores)))
;;         scores (reduce keep-score [] ops)]
;;     (apply + scores)))
;; (map cal-points [["5" "2" "C" "D" "+"] ["5" "-2" "4" "C" "D" "9" "+" "+"] ["1"]])

;; ;;693
;; (defn has-alternating-bits [n]
;;   (letfn [(alternating-bits? [bit n]
;;             (cond
;;               (= n 1) (not= bit n)
;;               (not= (bit-and n 1) bit) (alternating-bits? (bit-and n 1) (bit-shift-right n 1))
;;               :else false))]
;;     (alternating-bits? (bit-and n 1) (bit-shift-right n 1))))
;; (mapv has-alternating-bits [5 7 11 10 3 1])

;; ;;696
;; (defn count-binary-substrings [s]
;;   (let [cs (vec s)
;;         len (count s)
;;         count-bits (fn [[result cnt] index]
;;                      (cond
;;                        (= index len) (if (pos? cnt) [(conj result cnt) 0] [result 0])
;;                        (= (cs (dec index)) (cs index)) [result (inc cnt)]
;;                        :esle [(conj result cnt) 1]))
;;         bits-list (first (reduce count-bits [[] 1] (range 1 (inc len))))
;;         count-substrings (fn [s index]
;;                            (+ s (min (bits-list (dec index))
;;                                      (bits-list index))))]
;;     (reduce count-substrings 0 (range 1 (count bits-list)))))
;; (map count-binary-substrings ["00110011" "10101"])

;; ;;697
;; (defn find-shortest-sub-array [nums]
;;   (let [freqs (frequencies nums)
;;         max-count (apply max (vals freqs))
;;         most-frequent-nums (map first (filter (fn [[num cnt]] (= cnt max-count)) (into [] freqs)))
;;         get-subarray-length (fn [num]
;;                               (let [left (.indexOf nums num)
;;                                     right (.lastIndexOf nums num)]
;;                                 (inc (- right left))))]
;;     (apply min (map get-subarray-length most-frequent-nums))))
;; (map find-shortest-sub-array [[1,2,2,3,1] [1,2,2,3,1,4,2]])

;; ;; ;;704
;; (defn search [nums target]
;;   (letfn [(binary-search [nums target left right]
;;             (let [middle (quot (+ left right) 2)]
;;               (cond
;;                 (> left right) -1
;;                 (= (nums middle) target) middle
;;                 (> (nums middle) target) (binary-search nums target left (dec middle))
;;                 :else (binary-search nums target (inc middle) right))))]
;;     (binary-search nums target 0 (dec (count nums)))
;; ))
;; (map (partial apply search) ['([-1 0 3 5 9 12] 9) '([-1 0 3 5 9 12] 2)])

;; ;;709
;; (defn to-lower [s]
;;   (->> (map #(Character/toLowerCase %) (vec s))
;;        (str/join "")))
;; (map to-lower ["Hello" "here" "LOVELY"])

;; ;;717
;; (defn is-one-bit-character [bits]
;;   (letfn [(decode [bits]
;;             (cond
;;               (= bits [0]) true
;;               (= bits [1 0]) false
;;               (= bits [1 1]) false
;;               (zero? (first bits)) (decode (rest bits))
;;               :else (decode (drop 2 bits))
;;             ))]
;;     (decode bits)))
;; (map is-one-bit-character [[1 0 0] [1 1 1 0]])

;; ;; ;;724
;; (defn pivot-index [nums]
;;   (let [sum (apply + nums)
;;         find-pivot (fn [[left pivot-index] index]
;;                      (let [num (nums index)]
;;                        (if (= left (- sum num left))
;;                          (reduced [left index])
;;                          [(+ left num) pivot-index])))
;;         indices (range (count nums))]
;;     (last (reduce find-pivot [0 -1] indices))))
;; (map pivot-index [[1 7 3 6 5 6] [1 2 3] [2 1 -1]])

;; ;;728
;; (defn self-dividing-numbers [left right]
;;   (letfn [(integer-digits [n]
;;             (if (< n 10)
;;               [n]
;;               (conj (integer-digits (quot n 10))
;;                     (rem n 10))))
;;           (self-dividing-number? [n]
;;             (every? #(and (not= % 0) (zero? (rem n %))) (integer-digits n)))]
;;     (filter self-dividing-number? (range left (inc right)))))
;; (map (partial apply self-dividing-numbers) ['(1 22) '(47, 85)])

;; ;;733
;; (defn flood-fill [image sr sc new-color]
;;   (let [row (count image)
;;         col (count (image 0))
;;         new-image (make-array Long/TYPE row col)
;;         old-color ((image sr) sc)]
;;     (letfn [(valid? [[r c]]
;;               (and (>= r 0) (>= c 0) (< r row) (< c col)
;;                    (= (aget new-image r c) old-color)))
;;             (fill [r c]
;;               (let [indices [[(dec r) c] [(inc r) c] [r (dec c)] [r (inc c)]]]
;;                 (when (valid? [r c])
;;                   (do (aset new-image r c new-color)
;;                       (doseq [[r' c'] indices]
;;                         (fill r' c'))))))
;;             (clone-array []
;;               (doseq [r (range row) c (range col)]
;;                 (aset new-image r c ((image r) c))))]
;;       (clone-array)
;;       (fill sr sc)
;;       (mapv vec new-image))))
;; (map (partial apply flood-fill) ['([[1,1,1],[1,1,0],[1,0,1]] 1 1 2) '([[0,0,0],[0,0,0]] 0 0 2)])

;; ;;734
;; (defn are-sentences-similar [sentence1 sentence2 similar-pairs]
;;   (let [word-pair-map (reduce #(conj %1 (sort %2)) #{} similar-pairs)
;;         similar? (fn [[word1 word2]]
;;                    (contains? word-pair-map (sort [word1 word2])))]
;;     (cond
;;       (= sentence1 sentence2) true
;;       (not= (count sentence1) (count sentence2)) false
;;       :else (map similar? (map (comp sort vector) sentence1 sentence2)))))
;; (map (partial apply are-sentences-similar) ['(["great" "acting" "skills"] ["fine" "drama" "talent"] [["great" "fine"]  ["drama" "acting"] ["skills" "talent"]]) '(["great"] ["great"] []) '(["great"] ["doubleplus" "good"] [["great" "doubleplus"]])])

;; ;;744
;; (defn next-greatest-letter [letters target]
;;   (let [find-char (fn [result c]
;;                     (if (> (int c) (int target))
;;                       (reduced c)
;;                       result))]
;;     (reduce find-char (first letters) letters)))
;; (map (partial apply next-greatest-letter) ['([\c \f \j] \a) '([\c \f \j] \c) '([\c \f \j] \d) '([\c \f \j] \g) '([\c \f \j] \j)])

;; ;;746
;; (defn min-cost-climbing-stairs [cost]
;;   (let [end (count cost)]
;;     (letfn [(climb-stairs' [index]
;;               (+ (cost index)
;;                  (min (climb-stairs (inc index)) (climb-stairs (+ index 2)))))
;;             (climb-stairs [index]
;;               (if (< index end)
;;                 (climb-stairs' index)
;;                 0))]
;;       (min (climb-stairs 0) (climb-stairs 1)))))
;; (map min-cost-climbing-stairs [[10,15,20] [1,100,1,1,1,100,1,1,100,1]])

;; ;;747
;; (defn dominant-index [nums]
;;   (let [len (count nums)
;;         max-number (apply max nums)
;;         dominant (= (dec len) (count (filter #(>= max-number (* % 2)) nums)))]
;;     (if dominant
;;       (.indexOf nums max-number)
;;       -1)))
;; (map dominant-index [[3 6 1 0] [1 2 3 4] [1]])

;; (defn shortest-completing-word [license-plate words]
;;   (let [->letter-map (fn [s] (frequencies (vec (str/lower-case s))))
;;         plate-letter-map (->> (vec (str/lower-case license-plate))
;;                               (filter #(Character/isLetter %))
;;                               (frequencies))
;;         enough-character? (fn [letter-map key]
;;                             (let [cnt1 (get plate-letter-map key)
;;                                   cnt2 (get letter-map key)]
;;                               (and (not (or (nil? cnt1) (nil? cnt2)))
;;                                    (>= cnt2 cnt1))))
;;         completing-word? (fn [word]
;;                            (let [letter-map (->letter-map word)]
;;                              (every? #(enough-character? letter-map %) (keys plate-letter-map))))]
;;     (->> (filter completing-word? words)
;;          (sort (fn [l r] (compare (count l) (count r))))
;;          (first))))
;; (map (partial apply shortest-completing-word) ['("1s3 PSt" ["step","steps","stripe","stepple"])
;;                                                '("1s3 456" ["looks","pest","stew","show"])
;;                                                '("Ah71752" ["suggest","letter","of","husband","easy","education", "drug","prevent","writer","old"])
;;                                                '("OgEu755" ["enough","these","play","wide","wonder","box",        "arrive","money","tax","thus"])
;;                                                '("iMSlpe4" ["claim","consumer","student","camera","public",       "never","wonder","simple","thought","use"])])

;; ;;760
;; (defn anagram-mappings [nums1 nums2]
;;   (let [index-map (->> (map vector (range (count nums2)) nums2)
;;                        (reduce (fn [m [index num]]
;;                                  (let [indices (or (get m num) [])]
;;                                    (assoc m num (conj indices index))
;;                                  )) {})
;;                        )
;;         add-index (fn [[result index-map] num]
;;                     (let [indices (get index-map num)]
;;                         [(conj result (first indices)) (assoc index-map num (rest indices))]))]
;;    (first (reduce add-index [[] index-map] nums1))))
;; (map (partial apply anagram-mappings) ['([12 28 46 32 50] [50 12 32 46 28]) '([84 46] [84 46])
;;                                        '([84 46 46] [84 46 46])])

;; ;;762
;; (defn count-prime-set-bits [left right]
;;   (letfn [(count-set-bits [n]
;;             (let [bit (bit-and n 1)
;;                   n' (bit-shift-right n 1)]
;;               (if (< n 2)
;;                 n
;;                 (+ (count-set-bits n') bit))))
;;           (prime? [n]
;;             (let [factors (range 2 (inc (int (Math/ceil (Math/sqrt n)))))]
;;               (case n
;;                 0 false
;;                 1 false
;;                 2 true
;;                 (every? #(not (zero? (rem n %))) factors))))
;;           (prime-set-bits? [n]
;;             (prime? (count-set-bits n)))]
;;     (count (filter prime-set-bits? (range left (inc right))))))
;; (map (partial apply count-prime-set-bits) ['(6 10) '(10 15)])

;; ;;766
;; (defn is-teoplitz-matrix [matrix]
;;   (let [row (count matrix)
;;         col (count (matrix 0))
;;         indices (for [r (range (dec row)) c (range (dec col))]
;;                   [r c])
;;         compare-diagonal-elements (fn [result [r c]]
;;                                     (if (not= ((matrix r) c) ((matrix (inc r)) (inc c)))
;;                                       (reduced false)
;;                                       result))]
;;     (reduce compare-diagonal-elements true indices)))
;; (map is-teoplitz-matrix [[[1 2 3 4] [5 1 2 3] [9 5 1 2]] [[1 2] [2 2]]])

;; ;;771
;; (defn num-jewels-in-stones [jewels stones]
;;   (let [stone-count-map (frequencies stones)
;;         count-jewel (fn [sum jewel]
;;                       (+ sum (or (get stone-count-map jewel) 0)))]
;;     (reduce count-jewel 0 jewels)))
;; (map (partial apply num-jewels-in-stones) ['("aA" "aAAbbbb") '("z" "ZZ")])

;; ;;796
;; (defn rotate-string [s goal]
;;   (and (= (count s) (count goal))
;;        (str/includes? (str s s) goal)))
;; (map (partial apply rotate-string) ['("abcde" "cdeab") '("abcde" "abced")])

;; ;;800
;; (defn similar-rgb1 [color]
;;   (let [->decimal (fn [c]
;;                     (if (Character/isDigit c)
;;                       (- (int c) (int \0))
;;                       (- (+ (int c) 10) (int \a))))
;;         abs (fn [n] (if (neg? n) (- 0 n) n))
;;         from-digits (fn [digit1 digit2]
;;                       (+ (* digit1 16) digit2))

;;         similar-digits (fn [s]
;;                          (let [[digit1 digit2] (map ->decimal (vec s))
;;                                base-color (from-digits digit1 digit2)
;;                                compare-similarity (fn [[delta result] digit]
;;                                                     (let [value (abs (- (from-digits digit digit) base-color))]
;;                                                       (if (< value delta)
;;                                                         [value digit]
;;                                                         [delta result])))]
;;                            (last (reduce compare-similarity [base-color 0] (range 1 16)))))
;;         components (map #(subs color (inc (* 2 %)) (inc (* 2 (inc %)))) (range 3))
;;         ->hex (fn [h]
;;                 (if (> h 9)
;;                   (char (+ (int \a) (- h 10)))
;;                   (char (+ (int \0) h))))]
;;     (->> (map similar-digits components)
;;          (map (fn [h] (str (->hex h) (->hex h))))
;;          (str/join "")
;;          (str "#"))))
;; (defn similar-rgb [color]
;;   (let [colors (map #(str % %) (vec "0123456789abcdef"))
;;         abs (fn [n] (if (neg? n) (- 0 n) n))
;;         ->decimal (fn [s] (Integer/parseInt s 16))
;;         get-similarity (fn [s1 s2]
;;                          (abs (- (->decimal s1) (->decimal s2))))
;;         most-similar (fn [s]
;;                        (first (sort #(compare (get-similarity %1 s) (get-similarity %2 s)) colors)))
;;         color-components (map #(subs color % (+ % 2)) (range 1 6 2))]
;;     (map most-similar color-components)))
;; (map similar-rgb ["#09f166" "#4e3fe1"])

;; ;;804
;; (defn unique-morse-representations [words]
;;   (let [letters (vec "abcdefghijklmnopqrstuvwxyz")
;;         morse-codes [".-" "-..." "-.-." "-.." "." "..-." "--." "...." ".." ".---" "-.-" ".-.." "--" "-." "---" ".--." "--.-" ".-." "..." "-" "..-" "...-" ".--" "-..-" "-.--" "--.."]
;;         letter-morse-map (->> (map vector letters morse-codes)
;;                               (into {}))
;;         letter->morse #(get letter-morse-map %)
;;         transform (fn [word]
;;                     (str/join "" (map letter->morse (vec word))))]
;;     (count (set (map transform words)))))
;; (map unique-morse-representations [["gin" "zen" "gig" "msg"] ["a"]])

;; ;;806
;; (defn number-of-lines [widths s]
;;   (let [letters (map #(char (+ (int \a) %)) (range 26))
;;         letter-width-map (into {} (map vector letters widths))
;;         wrap-line (fn [[lines length] letter]
;;                     (let [width (get letter-width-map letter)]
;;                       (if (> (+ length width) 100)
;;                         [(inc lines) width]
;;                         [lines (+ length width)])))]
;;         (reduce wrap-line [1 0] (vec s))))
;; (map (partial apply number-of-lines)
;;         ['([10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10] "abcdefghijklmnopqrstuvwxyz")
;;          '([4 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10] "bbbcccdddaaa")])

;; ;;812
;; (defn largest-triangle-area [points]
;;   (let [len (count points)
;;         triangles (for [i (range (- len 2)) j (range (inc i) (dec len)) k (range (inc j) len)]
;;                     [(points i) (points j) (points k)])
;;         square (fn [n] (* n n))
;;         distance (fn [[x1 y1] [x2 y2]]
;;                    (Math/sqrt (+ (square (- x1 x2)) (square (- y1 y2)))))
;;         triangle-area (fn [[p1 p2 p3]]
;;                (let [a (distance p1 p2)
;;                      b (distance p2 p3)
;;                      c (distance p3 p1)
;;                      s (/ (+ a b c) 2)]
;;                  (Math/sqrt (* s (- s a) (- s b) (- s c)))))]
;;     (reduce (fn [max-area triangle]
;;               (let [area (triangle-area triangle)]
;;                 (max max-area area))) 0 triangles)))
;; (map largest-triangle-area [[[0 0] [0 1] [1 0] [0 2] [2 0]] [[1 0] [0 0] [0 1]]])

;; ;;819
;; (defn most-common-word [paragraph banned]
;;   (let [word-count-map (->> (str/lower-case paragraph)
;;                             (re-seq #"\w+")
;;                             (frequencies))
;;         remove-banned-word (fn [m word]
;;                              (if (nil? (get m word))
;;                                m
;;                                (dissoc m word)))
;;         common-words (into [] (reduce remove-banned-word word-count-map banned))
;;         compare-count (fn [l r]
;;           (compare (last r) (last l)))]
;;   (first (first (sort compare-count common-words)))))
;; (map (partial apply most-common-word) ['("Bob hit a ball, the hit BALL flew far after it was hit." ["hit"]) '("a." [])])

;; ;;821
;; (defn shortest-to-char1 [s c]
;;   (letfn [(get-indices [s c offset]
;;             (let [index (.indexOf (subs s offset) c)]
;;               (if (= index -1)
;;                 []
;;                 (vec (cons (+ offset index) (get-indices s c (+ offset (inc index))))))))]
;;     (let [len (count s)
;;           indices (get-indices s c 0)
;;           part1 (vec (reverse (range 1 (inc (first indices)))))
;;           part3 (vec (range (inc (last indices)) (- len (last indices))))
;;           generate-distances (fn [n]
;;                                (let [distances (make-array Long/TYPE (inc n))
;;                                      end (if (even? n)
;;                                            (quot n 2)
;;                                            (quot (inc n) 2))]
;;                                  (doseq [index (range end)]
;;                                    (aset distances index (inc index))
;;                                    (aset distances (- n 1 index) (inc index)))
;;                                  (vec distances)))
;;           add-distances (fn [result index]
;;                           (let [n (dec (- (indices index) (indices (dec index))))]
;;                             (concat result
;;                                    (generate-distances n)
;;                                     )))
;;           part2 (reduce add-distances [0] (range 1 (count indices)))
;;           ]
;;       (concat part1 part2 part3)
;;       )))

;; (defn shortest-to-char [s c]
;;   (let [abs (fn [n] (if (neg? n)
;;                       (- 0 n)
;;                       n))
;;         cs (vec s)
;;         len (count s)
;;         target (first (vec c))
;;         distances (make-array Long/TYPE len)
;;         target? (fn [index] (= (cs index) target))
;;         get-forward-distance (fn [pos index]
;;                                (if (target? index)
;;                                  0
;;                                  (- index pos)))
;;         get-backward-distance (fn [pos index]
;;                                 (if (target? index)
;;                                   0
;;                                   (min (aget distances index) (- pos index))))
;;         update-distance (fn [get-distance pos index]
;;                           (let [distance (get-distance pos index)]
;;                             (aset distances index distance)
;;                             (if (= (cs index) target)
;;                               index
;;                               pos)))
;;         pos (reduce (partial update-distance get-forward-distance) (- 0 len) (range len))]
;;     (reduce (partial update-distance get-backward-distance) pos (reverse (range 0 pos)))
;;     (vec distances)))
;; (map (partial apply shortest-to-char) ['("loveleetcode" "e") '("aaab" "b")])

;; ;;824
;; (defn to-goat-latin [sentence]
;;   (let [words (vec (re-seq #"\w+" sentence))
;;         vowel? (fn [c]
;;                  (contains? #{\a \e \i \o \u} (Character/toLowerCase c)))
;;         ->goat-latin-rule1 (fn [word]
;;                              (let [cs (vec word)]
;;                                (if (vowel? (first cs))
;;                                  (str word "ma")
;;                                  (str (subs word 1) (first cs) "ma"))))
;;         ->goat-latin-rule2 (fn [index word]
;;                              (apply str (cons word (repeat index "a"))))
;;         ->goat-latin (fn [index word]
;;                         (->goat-latin-rule2 index (->goat-latin-rule1 word)))]
;;     (->> (map ->goat-latin (range 1 (inc (count words))) words)
;;          (str/join " "))))
;; (map to-goat-latin ["I speak Goat Latin" "The quick brown fox jumped over the lazy dog"])

;; ;;830
;; (defn large-group-positions [s]
;;   (let [cs (vec s)
;;         add-large-group (fn [[groups start] index]
;;                           (cond
;;                             (= index (count s)) [(conj groups [start (dec index)]) index]
;;                             (= (cs index) (cs (dec index))) [groups start]
;;                             :else [(conj groups [start (dec index)]) index]))]
;;     (->> (reduce add-large-group [[] 0] (range 1 (inc (count s))))
;;          (first)
;;          (filter #(>= (- (last %) (first %)) 2)))))
;; (map large-group-positions ["abbxxxxzzy" "abc" "abcdddeeeeaabbbcd" "aba"])
;; ;;832
;; (defn flip-and-invert-image [image]
;;   (->> (map reverse image)
;;        (map #(map (fn [bit] (bit-xor bit 1)) %))
;;        )
;;   )
;; (map flip-and-invert-image [[[1 1 0] [1 0 1] [0 0 0]] [[1 1 0 0] [1 0 0 1] [0 1 1 1] [1 0 1 0]]])

;; ;;836
;; (defn is-rectangle-overlap [rec1 rec2]
;;   (let [abs (fn [n] (if (neg? n) (- 0 n) n))
;;         get-center (fn [[x1 y1 x2 y2]]
;;                      [(/ (+ x1 x2) 2) (/ (+ y1 y2) 2)])
;;         get-half-length (fn [[x1 y1 x2 y2]]
;;                           (/ (- x2 x1) 2))
;;         get-half-width (fn [[x1 y1 x2 y2]]
;;                          (/ (- y2 y1) 2))
;;         get-x-distance (fn [[x1 y1] [x2 y2]]
;;                          (abs (- x2 x1)))
;;         get-y-distance (fn [[x1 y1] [x2 y2]]
;;                          (abs (- y2 y1)))
;;         center1 (get-center rec1)
;;         center2 (get-center rec2)
;;         x-distance (get-x-distance center1 center2)
;;         y-distance (get-y-distance center1 center2)]
;;     (and (< x-distance (+ (get-half-length rec1) (get-half-length rec2)))
;;          (< y-distance (+ (get-half-width rec1) (get-half-width rec2))))
;;     ))
;; (map (partial apply is-rectangle-overlap) ['([0 0 2 2] [1 1 3 3]) '([0 0 1 1] [1 0 2 1]) '([0 0 1 1] [2 2 3 3])])

;; ;;844
;; (defn backspace-compare [s t]
;;   (let [type-char (fn [result c]
;;                     (cond
;;                       (and (= \# c) (empty? result)) result
;;                       (= \# c) (drop-last result)
;;                       :else (conj result c)))
;;         type-string (fn [s]
;;                       (reduce type-char [] (vec s)))]
;;     (= (type-string s) (type-string t))))
;; (map (partial apply backspace-compare) ['("ab#c" "ad#c") '("ab##" "c#d#") '("a##c" "#a#c") '("a#c" "b")])

;; ;;852
;; (defn peak-index-in-mountain-array1 [arr]
;;   (let [peak (apply max arr)]
;;     (.indexOf arr peak)))

;; (defn peak-index-in-mountain-array [arr]
;;   (let [peak-index? (fn [index]
;;                       (and (< (arr (dec index)) (arr index))
;;                            (> (arr index) (arr (inc index)))))
;;         find-peak-index (fn [result index]
;;                           (if (peak-index? index)
;;                             (reduced index)
;;                             result))]
;;     (reduce find-peak-index nil (range 1 (dec (count arr))))))
;; (map peak-index-in-mountain-array [[0 1 0] [0 2 1 0] [0 10 5 2] [3 4 5 1] [24 69 100 99 79 78 67 36 26 19]])

;; ;;859
;; (defn buddy-strings [s goal]
;;   (let [cs1 (vec s)
;;         cs2 (vec goal)]
;;     (cond
;;       (= cs1 cs2) (not= (count (set cs1)) (count cs1))
;;       (= (frequencies cs1) (frequencies cs2)) (= 2 (count (filter (fn [index] (not= (cs1 index) (cs2 index))) (range (count s)))))
;;       :else false)))
;; (map (partial apply buddy-strings) ['("ab" "ba") '("ab" "ab") '("aa" "aa") '("aaaaaaabc" "aaaaaaacb")])

;; ;;860
;; (defn lemonade-change [bills]
;;   (let [transact (fn [[result fives tens] bill]
;;                    (let [change (- bill 5)]
;;                      (case change
;;                        5 (if (pos? fives)
;;                            [result (dec fives) (inc tens)]
;;                            (reduced [false fives tens]))
;;                        15 (cond
;;                             (and (pos? fives) (pos? tens)) [result (dec fives) (dec tens)]
;;                             (>= fives 3) [result (- fives 3) tens]
;;                             :else (reduced [false fives tens]))
;;                        [result (inc fives) tens])))]
;;    (first (reduce transact [true 0 0] bills))))
;; (map lemonade-change [[5 5 5 10 20] [5 5 10 10 20] [5 5 10] [10 10]])

;; ;;867
;; (defn transpose [matrix]
;;   (let [row (count matrix)
;;         col (count (matrix 0))
;;         mt (make-array Long/TYPE col row)
;;         indices (for [r (range row) c (range col)]
;;                   [r c])]
;;     (doseq [[r c] indices]
;;       (aset mt c r ((matrix r) c)))
;;     (mapv vec mt)))
;; (map transpose [[[1 2 3] [4 5 6] [7 8 9]] [[1 2 3] [4 5 6]]])

;; ;;868
;; (defn binary-gap [n]
;;   (letfn [(integer-digits [n]
;;             (if (< n 2)
;;               [n]
;;               (conj (integer-digits (bit-shift-right n 1)) (bit-and n 1))))]
;;     (let [digits (integer-digits n)
;;           get-max-distance (fn [[max-distance start] index]
;;                              (cond
;;                                (and (= (digits index) 1) (nil? start)) [max-distance index]
;;                                (= (digits index) 1) [(max max-distance (- index start)) index]
;;                                :else [max-distance start]))]
;;       (first (reduce get-max-distance [0 nil] (range (count digits)))
;;       ))))
;; (map binary-gap [22 5 6 8 1])

;; ;;883
;; (defn projection-area1 [grid]
;;   (let [row (count grid)
;;         col (count (grid 0))
;;         indices (for [r (range row) c (range col)] [r c])
;;         sum (fn [xs] (apply + xs))
;;         transpose (fn [m]
;;                     (let [row (count m)
;;                           col (count (m 0))
;;                           mt (make-array Long/TYPE col row)
;;                           indices (for [r (range row) c (range col)]
;;                                     [r c])]
;;                       (doseq [[r c] indices]
;;                         (aset mt c r ((m r) c)))
;;                       (mapv vec mt)))
;;         top-area (count (filter (fn [[r c]] (pos? ((grid r) c))) indices))
;;         front-area (sum (map #(apply max %) grid))
;;         side-area (sum (map #(apply max %) (transpose grid)))]
;;     (+ top-area front-area side-area)))
;; (defn projection-area [grid]
;;   (let [row (count grid)
;;         col (count (grid 0))
;;         count-area (fn [cubes-area r]
;;                      (let [count-area' (fn [[area max-row max-col] c]
;;                                          (let [value1 ((grid r) c)
;;                                                value2 ((grid c) r)
;;                                                ]
;;                                            (if (pos? value1)
;;                                              [(inc area) (max value1 max-row) (max value2 max-col)]
;;                                              [area max-row max-col])))]
;;                       (apply + (reduce count-area' [cubes-area 0 0] (range col)))))]
;;     (reduce count-area 0 (range row))))
;; (map projection-area [[[1 2] [3 4]] [[2]] [[1 0] [0 2]] [[1 1 1] [1 0 1] [1 1 1]] [[2 2 2] [2 1 2] [2 2 2]]])

;; ;;884
;; ;; (defn uncommon-from-sentences [s1 s2]
;; ;;   (let [words (fn [s] (re-seq #"\w+" s))
;; ;;         unique-words (fn [s] (->> (into [] (frequencies (words s)))
;; ;;                                   (filter #(= (last %) 1))
;; ;;                                   (map first)))
;; ;;         words1 (unique-words s1)
;; ;;         words2 (unique-words s2)
;; ;;         set1 (set (concat words1 words2))
;; ;;         set2 (set/intersection (set words1) (set words2))]
;; ;;     (vec (set/difference set1 set2))))
;; ;; (map (partial apply uncommon-from-sentences) ['("this apple is sweet" "this apple is sour") '("apple apple" "banana")])

;; ;;888
;; (defn fair-candy-swap [alice-sizes bob-sizes]
;;   (let [s1 (apply + alice-sizes)
;;         s2 (apply + bob-sizes)
;;         freqs1 (frequencies alice-sizes)
;;         freqs2 (frequencies bob-sizes)
;;         delta (/ (- s1 s2) 2)]
;;     (reduce (fn [result size]
;;               (let [size' (- size delta)]
;;                 (if (nil? (get freqs2 size'))
;;                   result
;;                   [size size']))) [] (keys freqs1))))
;; (map (partial apply fair-candy-swap) ['([1 1] [2 2]) '([1 2] [2 3]) '([2] [1 3]) '([1 2 5] [2 4])])

;; ;;892
;; (defn surface-area [grid]
;;   (let [row (count grid)
;;         col (count (grid 0))
;;         height (apply max (flatten grid))
;;         cubes (make-array Long/TYPE row col height)
;;         not-cube? (fn [[r c h]] (or (< r 0) (>= r row) (>= c col) (>= h height) (< c 0) (< h 0) (= 0 (aget cubes r c h))))
;;         area (fn [r c h]
;;                (let [neighbours [[(inc r) c h] [(dec r) c h] [r (inc c) h] [r (dec c) h] [r c (inc h)] [r c (dec h)]]]
;;                  (count (filter not-cube? neighbours))))]
;;     (doseq [r (range row) c (range col) h (range ((grid r) c))]
;;       (aset cubes r c h 1))
;;     (->> (for [r (range row) c (range col) h (range ((grid r) c))]
;;            (area r c h))
;;          (flatten)
;;          (apply +))))
;; (map surface-area [[[2]] [[1 2] [3 4]] [[1 0] [0 2]] [[1 1 1] [1 0 1] [1 1 1]] [[2 2 2] [2 1 2] [2 2 2]]])

;; ;;896
;; (defn is-monotonic [nums]
;;   (let [len (count nums)
;;         not-increasing? (fn [nums index]
;;                           (> (nums (dec index)) (nums index)))
;;         monotone-increasing? (fn [nums] (reduce #(if (not-increasing? nums %2)
;;                                                    (reduced false)
;;                                                    %1)
;;                                                 true (range 1 (count nums))))]
;;     (if (< len 3)
;;       true
;;       (or (monotone-increasing? nums)
;;           (monotone-increasing? (vec (reverse nums)))))))
;; (map is-monotonic [[1 2 2 3] [6 5 4 4] [1 3 2] [1 2 4 5] [1 1 1]])

;; ;;905
;; (defn sort-array-by-partity [nums]
;;   (letfn [(sort-array [nums start end]
;;             (cond
;;               (>= start end) (vec nums)
;;               (and (odd? (aget nums start)) (even? (aget nums end))) (let [odd-number (aget nums start)
;;                                                                            even-number (aget nums end)]
;;                                                                        (aset nums start even-number)
;;                                                                        (aset nums end odd-number))
;;               :else (let [start' (if (even? (aget nums start)) (inc start) start)
;;                           end' (if (odd? (aget nums end)) (dec end) end)]
;;                       (sort-array nums start' end'))))]
;;     (let [xs (into-array nums)]
;;       (sort-array xs 0 (dec (count xs)))
;;       (vec xs))))
;; (map sort-array-by-partity [[3 1 2 4] [0]])

;; ;;908
;; (defn smallest-range-i [nums k]
;;   (let [min-value (apply min nums)
;;         max-value (apply max nums)]
;;       (max 0 (- max-value min-value (* 2 k)))))
;; (map (partial apply smallest-range-i) ['([1] 0) '([0 10] 2) '([1 3 6] 3)])

;; ;;914
;; (defn has-groups-size-x [deck]
;;   (letfn [(gcd [a b]
;;             (if (zero? (rem a b))
;;               b
;;               (gcd b (rem a b))))]
;;     (let [counts (vec (sort > (vals (frequencies deck))))]
;;       (< 1 (reduce gcd (first counts) (rest counts))))))
;; (map has-groups-size-x [[1 2 3 4 4 3 2 1] [1 1 1 2 2 2 3 3] [1] [1 1] [1 1 2 2 2 2]])

;; ;;917
;; (defn reverse-only-letters [s]
;;   (let [cs (vec s)
;;         index-letter-pairs (filter #(Character/isLetter (last %)) (map vector (range (count cs)) cs))
;;         indices (map first index-letter-pairs)
;;         letters (map last index-letter-pairs)
;;         index-letter-map (into {} (map vector indices (reverse letters)))
;;         get-letter (fn [index]
;;                      (or (get index-letter-map index) (cs index)))]
;;     (str/join "" (map get-letter (range (count cs))))))
;; (map reverse-only-letters ["ab-cd" "a-bC-dEf-ghIj" "Test1ng-Leet=code-Q!"])

;; ;; ;;922
;; ;; (defn sort-array-by-partity-ii [nums])
;; ;; (map sort-array-by-partity-ii [[4 2 5 7] [2 3]])

;; ;;925
;; (defn is-long-pressed-name1 [name typed]
;;   (let [pattern (re-pattern (str/join "+" (vec name)))]
;;   (= [typed] (re-seq pattern typed))))

;; (defn is-long-pressed-name1 [name typed]
;;   (let [group-chars (fn [s]
;;                       (let [group (fn [[result start] index]
;;                                     (cond
;;                                       (= index (count s)) [(conj result (subs s start)) index]
;;                                       (= (subs s (dec index) index) (subs s index (inc index))) [result start]
;;                                       :else [(conj result (subs s start index)) index]))]
;;                         (reduce group [[] 0] (range 1 (inc (count s))))))
;;         group1 (first (group-chars name))
;;         group2 (first (group-chars typed))
;;         same-or-long-pressed? (fn [result index]
;;                                 (if (str/includes? (group2 index) (group1 index))
;;                                   result
;;                                   (reduced false)))]
;;     (if (not= (count group1) (count group2))
;;       false
;;       (reduce same-or-long-pressed? true (range (count group1))))))

;; (defn is-long-pressed-name [name typed]
;;   (let [cs1 (vec name)
;;         cs2 (vec typed)]
;;   (letfn [(long-pressed-name? [index1 index2]
;;             (cond
;;               (>= index2 (count cs2)) (= index1 (count cs1))
;;               (= (cs1 index1) (cs2 index2)) (long-pressed-name? (inc index1) (inc index2))
;;               (and (pos? index1) (= (cs1 (dec index1)) (cs2 index2))) (long-pressed-name? index1 (inc index2))
;;               :else false))]
;;     (long-pressed-name? 0 0))))
;; (map (partial apply is-long-pressed-name) ['("alex" "aaleex") '("saeed" "ssaaedd") '("leelee" "lleeelee") '("laiden" "laiden")])

;; ;;929
;; (defn num-unique-emails [emails]
;;   (let [add-char (fn [result c]
;;                    (case c
;;                      \. result
;;                      \+ (reduced result)
;;                      (conj result c)))
;;         simplify (fn [email]
;;                    (let [[local-name domain-name] (str/split email #"@")

;;                          name (str/join "" (reduce add-char [] (vec local-name)))]
;;                      (str name "@" domain-name)))]
;;     (->> (map simplify emails)
;;          (distinct)
;;          (count))))
;; (map num-unique-emails [["test.email+alex@leetcode.com","test.e.mail+bob.cathy@leetcode.com","testemail+david@lee.tcode.com"] ["a@leetcode.com","b@leetcode.com","c@leetcode.com"]])

;; ;;937
;; (defn record-log-files [logs]
;;   (let [digit-log? #(Character/isDigit (last (vec %)))
;;         digit-logs (filter digit-log? logs)
;;         letter-logs (filter (complement digit-log?) logs)
;;         get-id (fn [log] (subs log 0 (.indexOf log " ")))
;;         get-content (fn [log] (subs log (inc (.indexOf log " "))))
;;         compare-letter-log (fn [l r]
;;                       (let [content1 (get-content l)
;;                             content2 (get-content r)]
;;                       (if (= content1 content2)
;;                         (compare (get-id l) (get-id r))
;;                         (compare content1 content2))))]
;;     (->> (sort compare-letter-log letter-logs)
;;          (#(concat % digit-logs))
;;     )))
;; (map record-log-files [["dig1 8 1 5 1" "let1 art can" "dig2 3 6" "let2 own kit dig" "let3 art zero"] ["a1 9 2 3 1" "g1 act car" "zo4 4 7" "ab1 off key dog" "a8 act zoo"]])

;; ;; ;;941
;; (defn valid-mountain-array1 [arr]
;;   (let [increasing? (fn [arr]
;;                       (reduce (fn [result index]
;;                                 (if (< (arr (dec index)) (arr index))
;;                                   result
;;                                   (reduced false)))
;;                               true (range 1 (count arr))))
;;         valid? (fn [arr]
;;                  (let [max-value (apply max arr)
;;                        index (.indexOf arr max-value)]
;;                    (and (> index 0) (< index (- (count arr) 1))
;;                         (increasing? (subvec arr 0 (inc index))) 
;;                         (increasing? (vec (reverse (subvec arr index)))))))]
;;   (if (< (count arr) 3)
;;     false
;;     (valid? arr)
;;   )))

;; (defn valid-mountain-array [arr]
;;   (let [get-left-peak-index (fn [arr]
;;                               (reduce (fn [result index]
;;                                         (if (< (arr (dec index)) (arr index))
;;                                           index
;;                                           (reduced result)
;;                                           )
;;                                         )  0 (range 1 (count arr)))
;;                               )
;;         get-right-peak-index (fn [arr]
;;                                (reduce (fn [result index]
;;                                          (if (< (arr (inc index)) (arr index))
;;                                            index
;;                                            (reduced result)
;;                                            )
;;                                          )
;;                                        (dec (count arr))
;;                                        (reverse (range 0 (dec (count arr))))
;;                                        )
;;                                )
;;         left-peak-index (get-left-peak-index arr)
;;         right-peak-index (get-right-peak-index arr)
;;         ]
;;     (and (> left-peak-index 0)
;;          (< right-peak-index (dec (count arr)))
;;          (= left-peak-index right-peak-index))))
;; (map valid-mountain-array [[2 1] [3 5 5] [0 3 2 1]])

;; ;;942
;; ;; (defn di-string-match [s]
;; ;;   (letfn [(match [cs left right]
;; ;;             (cond
;; ;;               (empty? cs) [left]
;; ;;               (= (first cs) \I) (cons left (match (rest cs) (inc left) right))
;; ;;               :else (cons right (match (rest cs) left (dec right)))))]
;; ;;     (let [n (count s)]
;; ;;       (match (vec s) 0 n))))
;; ;; (map di-string-match ["IDID" "III" "DDI"])

;; ;;944
;; (defn min-deletion-size [strs]
;;   (let [transpose (fn [m]
;;                     (let [row (count m)
;;                           col (count (m 0))
;;                           mt (make-array (type ((m 0) 0)) col row)
;;                           indices (for [r (range row) c (range col)]
;;                                     [r c])]
;;                       (doseq [[r c] indices]
;;                         (aset mt c r ((m r) c))
;;                         )
;;                       (mapv vec mt)))]
;;     (->> (mapv vec strs)
;;          (transpose)
;;          (filter #(not= % (sort %)))
;;          (count))))
;; (map min-deletion-size [["cba" "daf" "ghi"] ["a" "b"] ["zyx" "wvu" "tsr"]])

;; ;;953
;; (defn is-alien-sorted [words order]
;;   (let [dict (into {} (map vector (vec order) (vec "abcdefghijklmnopqrstuvwxyz")))
;;         translate-letter (fn [c]
;;                            (get dict c))
;;         translate (fn [s]
;;                          (str/join "" (map translate-letter (vec s))))
;;         compare-alien-word (fn [l r]
;;                              (compare (translate l) (translate r)))
;;         ]
;;     (= words (sort compare-alien-word words))))
;; (map (partial apply is-alien-sorted) ['(["hello","leetcode"] "hlabcdefgijkmnopqrstuvwxyz") '(["word","world","row"] "worldabcefghijkmnpqstuvxyz") '(["apple","app"] "abcdefghijklmnopqrstuvwxyz")])

;; ;;961
;; (defn repeated-n-times [nums]
;;   (let [find-repeated-number (fn [[num-set result] num]
;;               (if (contains? num-set num)
;;                 (reduced [num-set num])
;;                 [(conj num-set num) result]))]
;;    (last (reduce find-repeated-number [#{} nil] nums))))
;; (map repeated-n-times [[1 2 3 3] [2 1 2 5 3 2] [5 1 5 2 5 3 5 4]])

;; ;;976
;; (defn largest-perimeter [nums]
;;   (let [nums (vec (sort nums))
;;         len (count nums)
;;         triangles (distinct (for [i (range (- len 2)) j (range (inc i) (dec len)) k (range (inc j) len)]
;;                               [(nums i) (nums j) (nums k)]))
;;         triangle? (fn [[a b c]]
;;                     (and (> (+ a b) c) (> (+ b c) a) (> (+ c a) b)))
;;         perimeter (fn [[a b c]]
;;                     (if (triangle? [a b c])
;;                       (+ a b c)
;;                       0))]
;;     (apply max (map perimeter triangles))))
;; (map largest-perimeter [[2 1 2] [1 2 1] [3 2 3 4] [3 6 2 3]])

;; ;; ;;977
;; (defn sorted-squares1 [nums]
;;   (let [abs (fn [n] (if (neg? n) (- 0 n) n))
;;         square (fn [n] (* n n))
;;         compare-abs-value (fn [l r] (compare (abs l) (abs r)))]
;;    (->> (sort compare-abs-value nums)
;;         (map square))))

;; (defn sorted-squares [nums]
;;   (let [square (fn [n] (* n n))
;;         positive-num-index (reduce (fn [result index]
;;                                      (if (neg? (nums index))
;;                                        result
;;                                        (reduced index))) -1 (range (count nums)))]
;;     (letfn [(sorted-squares' [nums1 nums2]
;;               (cond
;;                 (empty? nums1) nums2
;;                 (empty? nums2) nums1
;;                 (> (first nums1) (first nums2)) (cons (first nums2) (sorted-squares' nums1 (rest nums2)))
;;                 (< (first nums1) (first nums2)) (cons (first nums1) (sorted-squares' (rest nums1) nums2))
;;                 :else (concat [(first nums1) (first nums2)] (sorted-squares' (rest nums1) (rest nums2)))))]
;;       (if (= positive-num-index -1)
;;         (map square (reverse nums))
;;         (sorted-squares' (map square (reverse (subvec nums 0 positive-num-index)))
;;                                (map square (subvec nums positive-num-index))))
;;       )))

;; (map sorted-squares [[-4,-1,0,3,10] [-7,-3,2,3,11]])

;; ;;989
;; (defn add-to-array-form [num k]
;;   (letfn [(integer-digits [n]
;;             (if (< n 10)
;;               [n]
;;               (conj (integer-digits (quot n 10)) (rem n 10))))]
;;     (let [digits1 (vec (reverse num))
;;           digits2 (vec (reverse (integer-digits k)))
;;           len1 (count digits1)
;;           len2 (count digits2)
;;           add-digit (fn [[digits carry] index]
;;                       (let [digit1 (if (< index len1) (digits1 index) 0)
;;                             digit2 (if (< index len2) (digits2 index) 0)
;;                             sum (+ digit1 digit2 carry)]
;;                         [(cons (rem sum 10) digits) (quot sum 10)]))
;;           add-digits (fn []
;;                        (let [[digits' carry] (reduce add-digit [[] 0] (range (max len1 len2)))]
;;                         (if (pos? carry)
;;                            (cons carry digits')
;;                            digits')
;;                          ))]
;;       (add-digits))))
;; (map (partial apply add-to-array-form) ['([1 2 0 0] 34) '([2 7 4] 181) '([2 1 5] 806) '([9 9 9 9 9 9 9 9 9 9] 1)])

;; ;;997
;; (defn find-judge1 [n trust]
;;   (let [persons (range 1 (inc n))
;;         judge-from-rule1 (set/difference (set persons) (set (map first trust)))
;;         trust-map (reduce (fn [m [person trusted-person]]
;;                                   (let [trusted-persons (or (get m person) [])]
;;                                     (assoc m person (conj trusted-persons trusted-person)))) {} trust)
;;         trusted-persons-list (map set (vals trust-map))
;;         judge-from-rule2 (apply set/intersection trusted-persons-list)]

;;     (if (and (= 1 (count judge-from-rule1)) (set/subset? judge-from-rule1 judge-from-rule2))
;;       (first judge-from-rule1)
;;       -1)))

;; (defn find-judge [n trust]
;;   (let [outgoing-relations (make-array Long/TYPE (inc n))
;;         ingoing-relations (make-array Long/TYPE (inc n))
;;         judge? (fn [index] (and (= 0 (aget outgoing-relations index))
;;                                 (= (dec n) (aget ingoing-relations index))))]
;;     (doseq [[person1 person2] trust]
;;       (aset outgoing-relations person1 (inc (aget outgoing-relations person1)))
;;       (aset ingoing-relations person2 (inc (aget ingoing-relations person2))))
;;     (reduce (fn [result index]
;;               (if (judge? index)
;;                (reduced index)
;;                 result)) -1 (range 1 (inc n)))))
;; (map (partial apply find-judge) ['(2 [[1 2]]) '(3 [[1 3] [2 3]]) '(3 [[1 3] [2 3] [3 1]]) '(3 [[1 2] [2 3]]) '(4 [[1 3] [1 4] [2 3] [2 4] [4 3]])])

;; ;; ;;999
;; (defn num-rook-captures [board]
;;   (let [row (count board)
;;         col (count (board 0))
;;         moves [[0 1] [0 -1] [1 0] [-1 0]]]
;;     (letfn [(find-rook []
;;               (reduce (fn [result r]
;;                         (let [c (.indexOf (board r) "R")]
;;                           (if (= c -1)
;;                             result
;;                             (reduced [r c])))) [] (range (count board))))
;;             (invalid? [r c] (not (and (>= r 0) (>= c 0) (< r row) (< c col)
;;                                (not= "B" ((board r) c)))))
;;             (attack [[r c] [delta-r delta-c]]
;;               (let [r' (+ r delta-r)
;;                     c' (+ c delta-c)]
;;                 (cond
;;                   (invalid? r' c') 0
;;                   (= "p" ((board r') c')) 1
;;                   :else (attack [r' c'] [delta-r delta-c]))))]
;;       (let [start-position (find-rook)]
;;        (apply + (map #(attack start-position %) moves))))))
;; (map num-rook-captures [[["." "." "." "." "." "." "." "."] ["." "." "." "p" "." "." "." "."] ["." "." "." "R" "."   "." "." "p"] ["." "." "." "." "." "." "." "."] ["." "." "." "." "." "." "." "."] ["." "." "." "p" "." "." "." "."] [". " "." "." "." "." "." "." "."] ["." "." "." "." "." "." "." "."]] [["." "." "." "." "." "." "." "."] ["." "p" "p" "p" "p" "p" "." "."] ["." "p" "p" "B"     "p" "p" "." "."] ["." "p" "B" "R" "B" "p" "." "."] ["." "p" "p" "B" "p" "p" "." "."] ["." "p" "p" "p" "p" "p" "." ".   "] ["." "." "." "." "." "." "." "."] ["." "." "." "." "." "." "." "."]] [["." "." "." "." "." "." "." "."] ["." "." "." "p" "." "." "." "."] ["." "." "." "p" ".  " "." "." "."] ["p" "p" "." "R" "." "p" "B" "."] ["." "." "." "." "." "." "." "."] ["." "." "." "B" "." "." "." "."]   ["." "." "." "p" "." "." "." "."] ["." "." "." "." "." "." "." "."]]])

;; ;;1002
;; (defn common-chars [words]
;;   (->> (map vec words)
;;        (map set)
;;        (apply set/intersection)
;;        (vec)))
;; (map common-chars [["bella" "label" "roller"] ["cool" "lock" "cook"]])

;; ;;1005
;; (defn largest-sum-after-k-negations1 [nums k]
;;   (let [non-positive-count (count (filter (comp not pos?) nums))
;;         nums (sort nums)
;;         negate-array (fn [nums]
;;                        (if (<= k non-positive-count)
;;                          (apply + (concat (map #(- 0 %) (take k nums)) (drop k nums)))
;;                          (let [nums' (concat (map #(- 0 %) (take non-positive-count nums)) (drop non-positive-count nums))
;;                                k' (- k non-positive-count)]
;;                            (if (even? k')
;;                              (apply + nums')
;;                              (- (apply + nums') (* 2 (apply min nums')))
;;                              ))))]
;;     (negate-array nums)))

;; (defn largest-sum-after-k-negations [nums k]
;;   (let [xs (into-array (sort nums))
;;         k' (reduce (fn [negations index]
;;               (if (and (pos? negations) (neg? (aget xs index)))
;;                 (do
;;                   (aset xs index (- 0 (aget xs index)))
;;                   (dec negations))
;;                 (reduced k)))
;;             k (range (count nums)))]
;;     (if (even? k')
;;       (apply + xs)
;;       (- (apply + xs) (* (apply min xs) 2)))))
;; (map (partial apply largest-sum-after-k-negations) ['([4 2 3] 1) '([3 -1 0 2] 3) '([2 -3 -1 5 -4] 2)])

;; ;;1009
;; (defn bitwise-complement [n]
;;   (letfn [(bitwise-complement-digits [n]
;;             (if (< n 2)
;;               [(bit-xor n 1)]
;;               (conj (bitwise-complement-digits (bit-shift-right n 1))
;;                     (bit-xor (rem n 2) 1))))
;;           (from-digits [digits]
;;             (reduce #(+ (* %1 2) %2) 0 digits))]
;;     (if (zero? n)
;;       1
;;       (from-digits (bitwise-complement-digits n)))))
;; (map bitwise-complement [5 7 10 0])

;; ;;1013
;; (defn can-three-parts-equal-sum [arr]
;;   (let [sum (apply + arr)
;;         check-sum (fn [[s parts] num]
;;           (let [sum' (* (inc parts) (/ sum 3))
;;                 parts' (if (= sum' (+ s num))
;;                          (inc parts)
;;                          parts)]
;;             [(+ s num) parts']))
;;        parts (last (reduce check-sum [0 0] arr))]
;;     (= parts 3)))
;; (map can-three-parts-equal-sum [[0 2 1 -6 6 -7 9 1 2 0 1] [0 2 1 -6 6 7 9 -1 2 0 1] [3 3 6 5 -2 2 5 1 -9 4]])

;; ;;1018
;; (defn prefixes-div-by-5 [nums]
;;   (let [divisible-by-5? (fn [n] (zero? (rem n 5)))
;;         digits nums
;;         check-prefix (fn [[results num] digit]
;;                         (let [num' (+ (* num 2) digit)]
;;                           [(conj results (divisible-by-5? num')) num']))]
;;     (first (reduce check-prefix [[] 0] digits))))
;; (map prefixes-div-by-5 [[0 1 1] [1 1 1] [0 1 1 1 1 1] [1 1 1 0 1]])

;; ;;1021
;; (defn remove-outer-parentheses [s]
;;   (let [cs (vec s)
;;         group-parentheses (fn [[results start pairs] index]
;;                             (let [c (cs index)]
;;                               (cond
;;                                 (= c \() [results start (inc pairs)]
;;                                 (zero? (dec pairs)) [(conj results (subs s start (inc index))) (inc index) 0]
;;                                 :else [results start (dec pairs)])))
;;         remove-parentheses (fn [s]
;;                              (subs s 1 (dec (count s))))]
;;     (->> (reduce group-parentheses [[] 0 0] (range (count cs)))
;;          (first)
;;          (map remove-parentheses)
;;          (str/join ""))))
;; (map remove-outer-parentheses ["(()())(())" "(()())(())(()(()))" "()()"])

;; ;;1025
;; (defn divisor-game [n]
;;   (even? n))
;; (map divisor-game [2 3 4])

;; ;;1030
;; (defn all-cells-dist-order [rows cols r-center c-center]
;;   (let [abs (fn [n] (if (neg? n)
;;                       (- 0 n)
;;                       n))
;;         get-distance (fn [[r c]]
;;                        (+ (abs (- r r-center)) (abs (- c c-center))))
;;         ->distance-coordinate (fn [coordinate] [(get-distance coordinate) coordinate])
;;         coordinates (for [r (range rows) c (range cols)] [r c])
;;         compare-distance (fn [l r] (compare (first l) (first r)))]
;;     (->> (map ->distance-coordinate coordinates)
;;          (sort compare-distance)
;;          (map last))))
;; (map (partial apply all-cells-dist-order) ['(1 2 0 0) '(2 2 0 1) '(2 3 1 2)])

;; ;;1037
;; (defn is-boomerang [points]
;;   (let [len (count points)
;;         boomerang? (fn [[[x1 y1] [x2 y2] [x3 y3]]]
;;                      (let [[u1 u2] [(- x2 x3) (- y2 y3)]
;;                            [v1 v2] [(- x1 x2) (- y1 y2)]]
;;                        (not= (* u1 v2) (* v1 u2))))
;;         triple-points-list (for [i (range (- len 2)) j (range (inc i) (dec len)) k (range (inc j) len)]
;;                              [(points i) (points j) (points k)])]
;;     (every? boomerang? triple-points-list)
;;   ))
;; (map is-boomerang [[[1 1] [2 3] [3 2]] [[1 1] [2 2] [3 3]]])

;; ;;1046
;; (defn last-stone-weight [stones]
;;   (letfn [(smash [stones]
;;             (let [y (first stones)
;;                   x (second stones)]
;;               (cond
;;                 (empty? stones) 0
;;                 (= (count stones) 1) (first stones)
;;                 (= x y) (smash (drop 2 stones))
;;                 :else (smash (sort > (conj (drop 2 stones) (- y x)))))))]
;;     (smash (sort > stones))))
;; (map last-stone-weight [[2 7 4 1 8 1] [1]])

;; ;;1047
;; (defn remove-duplicates [s]
;;   (letfn [(remove-duplicates' [s]
;;             (let [cs (vec s)
;;                   add-chars (fn [[results result] c]
;;                               (if (= (last result) c)
;;                                 [results (conj result c)]
;;                                 [(conj results result) [c]]))
;;                   results (apply conj (reduce add-chars [[] [(first cs)]] (rest cs)))
;;                   s' (->> (filter #(= 1 (count %)) results)
;;                    (flatten)
;;                    (str/join ""))]
;;               (if (= s s')
;;                 s
;;                 (remove-duplicates' s'))))]
;;     (remove-duplicates' s)))
;; (map remove-duplicates ["abbaca" "azxxzy"])

;; ;;1051
;; (defn height-checker [heights]
;;   (let [expected (vec (sort heights))]
;;   (count (filter #(not= (expected %) (heights %)) (range (count heights))))))
;; (map height-checker [[1,1,4,2,1,3] [5,1,2,3,4] [1,2,3,4,5]])

;; ;;1056
;; (defn confusing-number [n]
;;   (let [non-mirror-digits #{2 3 4 5 7}
;;         mirror-digits {0 0 1 1 6 9 8 8 9 6}
;;         digits (map #(- (int %) (int \0)) (vec (str n)))
;;         without-non-mirror-digits (zero?
;;                                    (count
;;                                     (set/intersection (set digits) non-mirror-digits)))
;;         not-same-number? (fn [digits]
;;                       (not= digits (map #(get mirror-digits %) digits)))]
;;     (and without-non-mirror-digits
;;          (not-same-number? digits))))
;; (map confusing-number [6 89 11 25])

;; ;;1064
;; (defn fixed-point [arr]
;;   (letfn [(fixed-point-index [nums left right]
;;             (let [middle (quot (+ left right) 2)
;;                   num (nums middle)]
;;               (cond
;;                 (> left right) -1
;;                 (> middle num) (fixed-point-index arr (inc middle) right)
;;                 (< middle num) (fixed-point-index arr left (dec middle))
;;                 :else middle)))]
;;     (fixed-point-index arr 0 (dec (count arr)))))
;; (map fixed-point [[-10 -5 0 3 7] [0 2 5 8 17] [-10 -5 3 4 7 9]])

;; ;;1065
;; (defn index-pairs [text words]
;;   (letfn [(get-word-indices [word offset]
;;             (let [s (subs text offset)
;;                   start (.indexOf s word)
;;                   end (+ start (dec (count word)))]
;;               (if (or (>= offset (count text)) (= start -1))
;;                 []
;;                 (cons [(+ start offset) (+ end offset)] (get-word-indices word (+ offset (inc start)))))))]
;;     (->> (map #(get-word-indices % 0) words)
;;          (reduce concat [])
;;          (sort))))
;; (map (partial apply index-pairs) ['("thestoryofleetcodeandme" ["story" "fleet" "leetcode"]) '("ababa" ["aba" "ab"])])

;; ;;1071
;; (defn gcd-of-strings1 [str1 str2]
;;   (letfn [(gcd [a b]
;;             (if (zero? (rem a b))
;;               b
;;               (gcd b (rem a b))))]
;;     (let [[s1 s2] (if (>= (count str1) (count str2))
;;                     [str1 str2]
;;                     [str2 str1])
;;           len (gcd (count s1) (count s2))
;;           repeat-str (fn [n s] (str/join "" (repeat n s)))
;;           common-divisor (subs s1 0 len)
;;           divisible-by-common-divisor (fn [s]
;;                                         (= s (repeat-str (quot (count s) len) common-divisor)))
;;           has-common-divisor? (fn [s1 s2]
;;                                 (and (= (subs s1 0 len) (subs s2 0 len))
;;                                      (every? divisible-by-common-divisor [s1 s2])))]
;;       (if (has-common-divisor? s1 s2)
;;         common-divisor""))))
;; (defn gcd-of-strings [str1 str2]
;;   (letfn [(gcd [a b]
;;             (if (zero? (rem a b))
;;               b
;;               (gcd b (rem a b))))]
;;     (if (= (str str1 str2) (str str2 str1))
;;       (subs str1 0 (gcd (count str1) (count str2)))
;;       "")))
;; (map (partial apply gcd-of-strings) ['("ABCABC" "ABC") '("ABABAB" "ABAB") '("LEET" "CODE") '("ABCDEF" "ABC")])

;; ;;1078
;; (defn find-occurrences [text fs ss]
;;   (let [words (vec (re-seq #"\w+" text))
;;         add-third-word (fn [result index]
;;                          (if (= (subvec words (- index 2) index) [fs ss])
;;                            (conj result (words index))
;;                            result))]
;;     (reduce add-third-word [] (range 2 (count words)))))
;; (map (partial apply find-occurrences) ['("alice is a good girl she is a good student" "a" "good") '("we will we will rock you" "we" "will")])

;; ;;1085
;; (defn sum-of-digits [nums]
;;   (letfn [(sum-digits [n]
;;             (if (< n 10)
;;               n
;;               (+ (sum-digits (quot n 10)) (rem n 10))))]
;;     (->> (apply min nums)
;;          (sum-digits)
;;          (#(rem (inc %) 2))
;;   )))
;; (map sum-of-digits [[34 23 1 24 75 33 54 8] [99 77 33 66 55]])

;; ;;1086
;; (defn high-five [items]
;;   (let [add-score (fn [score-map [id score]]
;;                     (let [scores (or (get score-map id) [])]
;;                       (assoc score-map id (conj scores score))))
;;         score-map (reduce add-score {} items)
;;         average-top-five (fn [id]
;;                            (->> (get score-map id)
;;                                 (sort >)
;;                                 (take 5)
;;                                 (apply +)
;;                                 (#(quot % 5))
;;                            ))]
;;     (map (fn [id] [id (average-top-five id)]) (keys score-map))))
;; (map high-five [[[1 91] [1 92] [2 93] [2 97] [1 60] [2 77] [1 65] [1 87] [1 100] [2 100] [2 76]] [[1 100] [7 100] [1 100] [7 100] [1 100] [7 100] [1 100] [7 100] [1 100] [7 100]]])

;; ;;1089
;; (defn duplicate-zeros [arr]
;;   (letfn [(count-zeroes [xs]
;;             (last (reduce (fn [[length cnt] i]
;;                             (cond
;;                               (and (< length (count xs)) (zero? (aget xs i))) [(+ length 2) (inc cnt)]
;;                               (and (< length (count xs)) (not (zero? (aget xs i)))) [(inc length) cnt]
;;                               :else (reduced [length cnt])))
;;                           [0 0] (range (count xs)))))
;;           (duplicate [xs left right]
;;             (cond
;;               (= left right) (vec xs)
;;               (zero? left) (vec xs)
;;               (zero? (aget xs left)) (do
;;                                        (aset xs right 0)
;;                                        (aset xs (dec right) 0)
;;                                        (duplicate xs (dec left) (- right 2)))
;;               :else (do
;;                       (aset xs right (aget xs left))
;;                       (duplicate xs (dec left) (dec right)))))]
;;     (let [xs (into-array arr)
;;           cnt (count-zeroes xs)
;;           len (count xs)
;;           left (- len 1 cnt)
;;           right (dec len)]
;;       (duplicate xs left right))))
;; (map duplicate-zeros [[1 0 2 3 0 4 5 0] [1 2 3]])

;; ;;1099
;; (defn two-sum-less-than-k [nums k]
;;   (let [exists? (fn [target [result num-map] num]
;;                   (if (nil? (get num-map num))
;;                     [result (assoc num-map (- target num) num)]
;;                     (reduced [true num-map])))
;;         two-sum (fn [max-sum target]
;;                   (if (first (reduce (partial exists? target) [false {}] nums))
;;                     (reduced target)
;;                     max-sum))]
;;     (reduce two-sum -1 (reverse (range (apply min nums) k)))))
;; (map (partial apply two-sum-less-than-k) ['([34 23 1 24 75 33 54 8] 60) '([10 20 30] 15)])

;; ;;1103
;; (defn distribute-candies [candies num-people]
;;   (let [xs (make-array Long/TYPE num-people)]
;;     (letfn [(distribute [candies given]
;;               (aset xs (rem (dec given) num-people) (min candies given))
;;               (when (< given candies)
;;                   (distribute (- candies given) (inc given))))]
;;       (distribute candies 1)
;;       (vec xs))))
;; (map (partial apply distribute-candies) ['(7 4) '(10 3)])

;; ;;1108
;; (defn defang-ip-addr [address]
;;   (->> (str/split address #"\.")
;;        (str/join "[.]")))
;; (map defang-ip-addr ["1.1.1.1" "255.100.50.0"])

;; ;;1118
;; (defn number-of-days [year month]
;;   (let [month-days [31 28 31 30 31 30 31 31 30 31 30 31]
;;         divisible? (fn [a b] (zero? (rem a b)))
;;         leap-year? (fn [year] (or (and (divisible? year 4) (not (divisible? year 100)))
;;                                   (divisible? year 400)))
;;         leap-day (if (and (leap-year? year) (= month 2)) 1 0)]
;;     (+ (month-days (dec month))
;;        leap-day)))
;; (map (partial apply number-of-days) ['(1992 7) '(2000 2) '(1900 2)])

;; ;;1119
;; (defn remove-vowels [s]
;;   (let [non-vowel? (fn [c]
;;                      (not (contains? #{\a \e \i \o \u} c)))]
;;     (->> (vec s)
;;          (filter non-vowel?)
;;          (str/join ""))))
;; (map remove-vowels ["leetcodeisacommunityforcoders" "aeiou"])

;; ;;1122
;; (defn relative-sort-array [arr1 arr2]
;;   (let [freqs (frequencies arr1)
;;         repeat-num (fn [num]
;;                      (let [cnt (get freqs num)]
;;                        (take cnt (cycle [num]))))
;;         part1 (flatten (map repeat-num arr2))
;;         arr2-set (set arr2)
;;         not-in-arr2? (fn [num]
;;                        (not (contains? arr2-set num)))
;;         part2 (sort (filter not-in-arr2? arr1))]
;;     (concat part1 part2)))
;; (map (partial apply relative-sort-array) ['([2,3,1,3,2,4,6,7,9,2,19] [2,1,4,3,9,6]) '([28,6,22,8,44,17] [22,28,8,6])])

;; ;;1128
;; (defn num-equiv-domino-pairs [dominoes]
;;   (->> (map sort dominoes)
;;        (frequencies)
;;        (vals)
;;        (map (fn [n] (quot (* n (dec n)) 2)))
;;        (apply +)))
;; (map num-equiv-domino-pairs [[[1,2],[2,1],[3,4],[5,6]] [[1,2],[1,2],[1,1],[1,2],[2,2]]])

;; ;;1133
;; (defn largest-unique-number [nums]
;;   (let [max' #(if (empty? %) -1 (apply max %))]
;;   (->> (frequencies nums)
;;        (into)
;;        (filter #(= 1 (last %)))
;;        (map first)
;;        (max'))))
;; (map largest-unique-number [[5,7,3,9,4,9,8,3,1] [9,9,8,8]])

;; ;;1134
;; (defn is-armstrong [n]
;;   (let [integer-digits (fn [n]
;;                          (->> (vec (str n))
;;                               (map #(- (int %) (int \0)))))
;;         pow (fn [base exponent] (int (Math/pow base exponent)))
;;         armstrong-number? (fn [n]
;;                             (let [digits (integer-digits n)
;;                                   len (count digits)]
;;                               (= n (apply + (map #(pow % len) digits)))))]
;;     (armstrong-number? n)))
;; (map is-armstrong [153 123])

;; ;;1137
;; (defn tribonacci [n]
;;   (let [next-tribonacci-sequence (fn [[t0 t1 t2] _]
;;                           [t1 t2 (+ t0 t1 t2)])
;;         tribonacci' (fn [n]
;;                       (last
;;                        (reduce next-tribonacci-sequence [0 1 1] (range (- n 2)))))]
;;     (case n
;;       0 0
;;       1 1
;;       2 1
;;       (tribonacci' n))))
;; (map tribonacci [4 25])

;; ;;1150
;; (defn is-majority-element1 [nums target]
;;   (let [left (.indexOf nums target)
;;         right (.lastIndexOf nums target)
;;         len (count nums)]
;;     (and (>= left 0) (>= right 0) (>= (- right left) (quot len 2)))))
;; (defn is-majority-element [nums target]
;;   (let [left-index (.indexOf nums target)
;;         right-index (+ left-index (quot (count nums) 2))]
;;     (and (>= left-index 0) (< right-index (count nums)) (= target (nums right-index)))))
;; (map (partial apply is-majority-element) ['([2 4 5 5 5 5 5 6 6] 5) '([10 100 101 101] 101)])

;; ;;1154
;; (defn day-of-year [date]
;;   (let [month-days [31 28 31 30 31 30 31 31 30 31 30 31]
;;         divisible? (fn [a b] (zero? (rem a b)))
;;         leap-year? (fn [year] (or (and (divisible? year 4) (not (divisible? year 100)))
;;                                   (divisible? year 400)))
;;         [year month day] (vec (map #(Integer/parseInt %) (str/split date #"\-")))
;;         leap-day (if (and (leap-year? year) (> month 2)) 1 0)
;;         days (reduce #(+ %1 (month-days (dec %2))) 0 (range 1 month))]
;;     (+ days day leap-day)
;; ))
;; (map day-of-year ["2019-01-09" "2019-02-10" "2003-03-01" "2004-03-01"])

;; ;;1160
;; (defn count-characters [words chars]
;;   (let [freqs1 (frequencies (vec chars))
;;         check-char-count (fn [result [c cnt]]
;;                            (if (<= cnt (or (get freqs1 c) 0))
;;                              result
;;                              (reduced false)))
;;         good? (fn [s]
;;                 (->> (into [] (frequencies (vec s)))
;;                      (reduce check-char-count true)))]
;;     (->> (filter good? words)
;;          (map count)
;;          (apply +))))
;; (map (partial apply count-characters) ['(["cat" "bt" "hat" "tree"] "atach") '(["hello" "world" "leetcode"] "welldonehoneyr")])

;; ;;1165
;; (defn calculate-time [keyboard word]
;;   (let [letter-map (into {} (map vector (vec keyboard) (range (count keyboard))))
;;         abs (fn [n] (if (neg? n) (- 0 n) n))
;;         get-time (fn [[total prev] index]
;;                    (let [time (abs (- index prev))]
;;                      [(+ total time) index]))
;;         indices (map #(letter-map %) (vec word))
;;         ]
;;    (first (reduce get-time [0 0] indices))))
;; (map (partial apply calculate-time) ['("abcdefghijklmnopqrstuvwxyz" "cba") '("pqrstuvwxyzabcdefghijklmno" "leetcode")])

;; ;; ;;1175
;; (defn num-prime-arrangements [n]
;;   (let [prime? (fn [n]
;;                  (reduce (fn [result num]
;;                            (if (zero? (rem n num))
;;                              (reduced false)
;;                              result)) true (range 2 n)))
;;         count-primes (fn [n] (count (filter prime? (range 2 (inc n)))))
;;         modulo (fn [n] (rem n 1000000007))
;;         factorial (fn [n] (if (= n 0) 1
;;                               (reduce #(modulo (* %1 %2)) (range 1 (inc n)))))
;;         num-of-primes (count-primes n)]
;;     (modulo (* (factorial num-of-primes) (factorial (- n num-of-primes))))))
;; (map num-prime-arrangements [5 100])

;; ;;1176
;; (defn diet-plan-performance [calories k lower upper]
;;   (let [calculate-point (fn [total-calories]
;;                           (cond
;;                             (< total-calories lower) -1
;;                             (> total-calories upper) 1
;;                             :else 0))
;;         add-point (fn [[points prev-calories] index]
;;                     (let [calory0 (calories (- index k))
;;                           calory1 (calories index)
;;                           current-calories (+ (- prev-calories calory0) calory1)
;;                           point (calculate-point current-calories)]
;;                       [(+ points point) current-calories]))
;;         initial-calories (apply + (take k calories))
;;         initial-points-calories [(calculate-point initial-calories) initial-calories]]
;;     (first (reduce add-point initial-points-calories (range k (count calories))))))
;; (map (partial apply diet-plan-performance) ['([1 2 3 4 5] 1 3 3) '([3 2] 2 0 1) '([6 5 0 0] 2 1 5)])

;; ;;1180
;; (defn count-letters [s]
;;   (let [group-by-letter (fn [s]
;;                           (let [cs (vec s)
;;                                 add (fn [[strs start] index]
;;                                       (cond
;;                                         (= index (count s)) [(conj strs (subs s start index)) index]
;;                                         (= (cs index) (cs (dec index))) [strs start]
;;                                         :else [(conj strs (subs s start index)) index]))]
;;                          (first (reduce add [[] 0] (range 1 (inc (count s)))))))
;;         count-distinct-substr (fn [s]
;;                                 (apply + (range 1 (inc (count s)))))]
;;     (->> (group-by-letter s)
;;          (map count-distinct-substr)
;;          (apply +))))
;; (map count-letters ["aaaba" "aaaaaaaaaa"])

;; ;;1184
;; (defn distance-between-bus-stops [distance start destination]
;;   (let [len (count distance)
;;         get-distance (fn [src dst]
;;                        (let [stops (if (<= src dst)
;;                                      (range src dst)
;;                                      (range src (+ dst len)))
;;                              distances (map #(distance (rem % len)) stops)]
;;                          (apply + distances)))]
;;     (min (get-distance start destination) (get-distance destination start))))
;; (map (partial apply distance-between-bus-stops) ['([1,2,3,4] 0 1) '([1 2 3 4] 0 2) '([1 2 3 4] 0 3)])

;; ;;1185
;; (defn day-of-the-week [day month year]
;;   (let [day-names ["Sunday"  "Monday"  "Tuesday"  "Wednesday"  "Thursday"  "Friday"  "Saturday"]
;;         month-days [31 28 31 30 31 30 31 31 30 31 30 31]
;;         divisible? (fn [a b] (zero? (rem a b)))
;;         leap-year? (fn [year] (or (and (divisible? year 4) (not (divisible? year 100)))
;;                                   (divisible? year 400)))
;;         leap-day (if (and (> month 2) (leap-year? year))
;;                    1
;;                    0)
;;         add-year-days (fn [s year]
;;                         (let [days (if (leap-year? year) 366 365)]
;;                         (+ s days)))
;;         part1 (reduce add-year-days 0 (range 1970 year))
;;         part2 (+ (apply + (map #(month-days (dec %1)) (range 1 month))) leap-day)
;;         part3 day
;;         days (+ part1 part2 part3)]
;;     (day-names (rem (+ days 3) 7))))
;; (map (partial apply day-of-the-week) ['(1 1 1970)' (31 8 2019) '(18 7 1999) '(15 8 1993)])

;; ;;1189
;; (defn max-number-of-balloons [text]
;;   (let [balloon-freqs (frequencies (vec "balloon"))
;;         freqs (frequencies (vec text))
;;         get-num-of-ballons (fn [letter]
;;            (let [cnt1 (get balloon-freqs letter)
;;                  cnt2 (get freqs letter)]
;;              (if (nil? cnt2)
;;                0
;;                (quot cnt2 cnt1))))]
;;    (apply min (map get-num-of-ballons (keys balloon-freqs)))))
;; (map max-number-of-balloons ["nlaebolko" "loonbalxballpoon" "leetcode"])

;; ;;1196
;; (defn max-number-of-apples [weight]
;;   (let [weight (sort weight)
;;         count-apple (fn [[num-of-apples sum] w]
;;                       (if (<= (+ sum w) 5000)
;;                         [(inc num-of-apples) (+ sum w)]
;;                         (reduced [num-of-apples sum])))]
;;    (first (reduce count-apple [0 0] weight))))
;; (map max-number-of-apples [[100 200 150 1000] [900 950 800 1000 700 800]])

;; ;;1200
;; (defn minimum-abs-difference [arr]
;;   (let [xs (vec (sort arr))
;;         abs (fn [n] (if (neg? n) (- 0 n) n))
;;         ->pair-and-difference (fn [index]
;;                                 (let [x1 (xs (dec index))
;;                                       x2 (xs index)]
;;                                   [[x1 x2] (- x2 x1)]))
;;         pair-differences (map ->pair-and-difference (range 1 (count xs)))
;;         min-difference (apply min (map last pair-differences))
;;         min-difference? #(= (last %) min-difference)]
;;     (map first (filter min-difference? pair-differences))))
;; (map minimum-abs-difference [[4,2,1,3] [1,3,6,10,15] [3,8,-10,23,19,-4,-14,27]])

;; ;;1207
;; (defn unique-occurences [arr]
;;   (let [counts (vals (frequencies arr))]
;;     (= (count (distinct counts)) (count counts))))
;; (map unique-occurences [[1 2 2 1 1 3] [1 2] [-3 0 1 -3 1 1 1 -3 10 0]])

;; ;;1213
;; (defn arrays-intersection [arr1 arr2 arr3]
;;   (let [len1 (count arr1)
;;         len2 (count arr2)
;;         len3 (count arr3)
;;         append (fn [[results index1 index2 index3] _]
;;                  (if (or (>= index1 len1) (>= index2 len2) (>= index3 len3))
;;                    [results index1 index2 index3]
;;                    (let [num1 (arr1 index1) num2 (arr2 index2) num3 (arr3 index3)
;;                          max-num (max num1 num2 num3)
;;                          inc-index (fn [num index] (if (< num max-num) (inc index) index))]
;;                      (if (= num1 num2 num3)
;;                        [(conj results num1) (inc index1) (inc index2) (inc index3)]
;;                        [results (inc-index num1 index1) (inc-index num2 index2) (inc-index num3 index3)]))))]
;;  (first (reduce append [[] 0 0 0] (range (+ len1 len2 len3))))))

;; (defn arrays-intersection [arr1 arr2 arr3]
;;   (let [sets (map set [arr1 arr2 arr3])]
;;       (sort (vec (reduce #(set/intersection %2 %1) (first sets) (rest sets))))))
;; (map (partial apply arrays-intersection) ['([1 2 3 4 5] [1 2 5 7 9] [1 3 4 5 8]) '([197 418 523 876 1356] [501 880 1593 1710 1870] [521 682 1337 1395 1764])])

;; ;; ;;1217
;; (defn min-cost-to-move-chips1 [position]
;;   (let [freqs (frequencies position)
;;         positions (keys freqs)
;;         odd-indexed-nums (map #(get freqs %) (filter odd? positions))
;;         even-indexed-nums (map #(get freqs %) (filter even? positions))
;;         total (fn [xs] (apply + xs))]
;;     (min (total odd-indexed-nums) (total even-indexed-nums))))
;; (defn min-cost-to-move-chips [position]
;;   (let [count-even-odd (fn [[odds evens] index]
;;                           (if (odd? index)
;;                             [(inc odds) evens]
;;                             [odds (inc evens)]))
;;        [odds evens] (reduce count-even-odd [0 0] position)]
;;     (min odds evens)))
;; (map min-cost-to-move-chips [[1 2 3] [2 2 2 3 3] [1 1000000000]])

;; ;;1221
;; (defn balanced-string-split [s]
;;   (let [count-balanced-string (fn [[sum lefts rights] c]
;;                                 (let [lefts' (+ lefts (if (= c \L) 1 0))
;;                                       rights' (+ rights (if (= c \R) 1 0))]
;;                                   (if (= lefts' rights')
;;                                     [(inc sum) lefts' rights']
;;                                     [sum lefts' rights'])))]
;;    (first (reduce count-balanced-string [0 0 0] (vec s)))))
;; (map balanced-string-split ["RLRRLLRLRL" "RLLLLRRRLR" "LLLLRRRR" "RLRRRLLRLL"])

;; ;;1228
;; (defn missing-number [arr]
;;   (let [delta (/ (- (last arr) (first arr)) (count arr))
;;         find-missing-number (fn [result index]
;;                               (if (= (+ delta (arr (dec index))) (arr index))
;;                                 result
;;                                 (reduced (- (arr index) delta))))
;;         ]
;;     (reduce find-missing-number nil (range 1 (count arr)))))
;; (map missing-number [[5 7 11 13] [15 13 12]])

;; ;;1232
;; (defn check-straight-line [coordinates]
;;   (let [straight-line? (fn [[u1 u2] [v1 v2]]
;;                          (= (* u1 v2) (* u2 v1)))
;;         coordinates->vector' (fn [[x1 y1] [x2 y2]]
;;                                [(- x2 x1) (- y2 y1)])
;;         coordinates->vector (fn [index] (coordinates->vector' (coordinates (dec index)) (coordinates index)))
;;         vectors (mapv coordinates->vector (range 1 (count coordinates)))
;;         check-line (fn [result index]
;;           (if (straight-line? (vectors (dec index)) (vectors index))
;;             result
;;             (reduced false)))]
;;     (reduce check-line true (range 1 (count vectors)))))
;; (map check-straight-line [[[1 2] [2 3] [3 4] [4 5] [5 6] [6 7]] [[1 1] [2 2] [3 4] [4 5] [5 6] [7 7]]])

;; ;;1243
;; (defn transform-array [arr]
;;   (letfn [(transform [arr]
;;             (let [adjust-element (fn [results index]
;;                                    (let [x0 (arr (dec index))
;;                                          x1 (arr index)
;;                                          x2 (arr (inc index))]
;;                                      (cond
;;                                        (and (< x1 x0) (< x1 x2)) (conj results (inc x1))
;;                                        (and (> x1 x0) (> x1 x2)) (conj results (dec x1))
;;                                        :else (conj results x1))))
;;                   results (conj (reduce adjust-element [(first arr)] (range 1 (dec (count arr)))) (last arr))]
;;                   (if (not= results arr)
;;                     (transform results)
;;                     results)))]
;;     (transform arr)))
;; (map transform-array [[6 2 3 4] [1 6 3 4 3 5]])

;; ;;1252
;; (defn odd-cells1 [m n indices]
;;   (let [matrix (make-array Long/TYPE m n)
;;         inc-row-cells (fn [r]
;;                         (doseq [c (range n)]
;;                           (aset matrix r c (inc (aget matrix r c)))))
;;         inc-col-cells (fn [c]
;;                         (doseq [r (range m)]
;;                           (aset matrix r c (inc (aget matrix r c)))))
;;         inc-cells (fn [r c]
;;                     (inc-row-cells r)
;;                     (inc-col-cells c))
;;         all-indices (for [r (range m ) c (range n)] [r c])
;;         count-odd (fn [[r c]] (if (odd? (aget matrix r c)) 1 0))]
;;     (doseq [[r c] indices]
;;       (inc-cells r c))
;;     (->> (map count-odd all-indices)
;;          (apply +))))

;; (defn odd-cells [m n indices]
;;   (let [odd-rows (make-array Long/TYPE m)
;;         odd-cols (make-array Long/TYPE n)
;;         all-indices (for [r (range m) c (range n)]
;;                       [r c])
;;         odd-cell? (fn [[r c]]
;;                     (bit-xor (aget odd-rows r) (aget odd-cols c)))]
;;     (doseq [[r c] indices]
;;         (aset odd-rows r (bit-xor (aget odd-rows r) 1))
;;         (aset odd-cols c (bit-xor (aget odd-cols c) 1)))
;;     (apply + (map odd-cell? all-indices))))
;; (map (partial apply odd-cells) ['(2 3 [[0 1] [1 1]]) '(2 2 [[1 1] [0 0]])])

;; ;;1260
;; (defn shift-grid [grid k]
;;   (let [row (count grid)
;;         col (count (grid 0))
;;         matrix (make-array Long/TYPE row col)
;;         rule1 (fn [grid]
;;                 (doseq [r (range row) c (range (dec col))]
;;                   (aset matrix r (inc c) ((grid r) c))))
;;         rule2 (fn [grid]
;;                 (doseq [r (range (dec row))]
;;                   (aset matrix (inc r) 0 ((grid r) (dec col)))))
;;         rule3 (fn [grid]
;;                 (aset matrix 0 0 ((grid (dec row)) (dec col))))
;;         shift' (fn [grid]
;;                  ((juxt rule1 rule2 rule3) grid)
;;                  (mapv vec matrix))
;;         shift (fn [grid _] (shift' grid))]
;;     (reduce shift grid (range k))))
;; (map (partial apply shift-grid) ['([[1 2 3] [4 5 6] [7 8 9]] 1) '([[3 8 1 9] [19 7 2 5] [4 6 11 10] [12 0 21 13]] 4) '([[1 2 3] [4 5 6] [7 8 9]] 9)])

;; ;;1266
;; (defn min-time-to-visit-all-points [points]
;;   (let [abs (fn [n] (if (neg? n) (- n) n))
;;         min-time (fn [index]
;;                    (let [[x1 y1] (points (dec index))
;;                          [x2 y2] (points index)]
;;                      (max (abs (- x2 x1)) (abs (- y2 y1)))))
;;         sum-min-time #(+ %1 (min-time %2))]
;;     (reduce sum-min-time 0 (range 1 (count points)))))
;; (map min-time-to-visit-all-points [[[1 1] [3 4] [-1 0]] [[3 2] [-2 2]]])

;; ;;1271
;; (defn to-hexspeak [num]
;;   (letfn [(->hex-digits [n]
;;             (if (< n 16)
;;               [n]
;;               (conj (->hex-digits (quot n 16)) (rem n 16))))
;;           (only-1-or-0? [digits]
;;             (zero? (count (filter #(> % 1) digits))))
;;           (->hexspeak [digits]
;;             (if (only-1-or-0? digits)
;;               (str/join "" (map #(if (zero? %) \O \I) digits))
;;               "ERROR"))]
;;     (->> (Integer/parseInt num)
;;          (->hex-digits)
;;          (->hexspeak))))
;; (map to-hexspeak ["257" "3"])

;; ;;1275
;; (defn tictactoe [moves]
;;   (let [row 3
;;         col 3
;;         matrix (make-array Long/TYPE row col)
;;         get-row-winner (fn [matrix]
;;                          (reduce (fn [winner r]
;;                                    (if (and (pos? (aget matrix r 0)) (apply = (aget matrix r)))
;;                                      (reduced (aget matrix r 0))
;;                                      winner)) nil (range row)))

;;         get-column-winner (fn [matrix]
;;                             (reduce (fn [winner c]
;;                                       (if (and (pos? (aget matrix 0 c)) (= (aget matrix 0 c) (aget matrix 1 c) (aget matrix 2 c)))
;;                                         (reduced (aget matrix 0 c))
;;                                         winner))
;;                                     nil (range col)))
;;         get-diagonal-winner (fn [matrix]
;;                               (cond
;;                                 (and (pos? (aget matrix 1 1)) (= (aget matrix 0 0) (aget matrix 1 1) (aget matrix 2 2))) (aget matrix 1 1)
;;                                 (and (pos? (aget matrix 1 1)) (= (aget matrix 0 2) (aget matrix 1 1) (aget matrix 2 0))) (aget matrix 1 1)
;;                                 :else nil))
;;         get-winner (fn [matrix]
;;                      (or (get-column-winner matrix) (get-row-winner matrix) (get-diagonal-winner matrix)))
;;         pending? (fn []
;;                    (let [indices (for [r (range row) c (range col)]
;;                                    [r c])]
;;                      (reduce (fn [result [r c]]
;;                                (if (zero? (aget matrix r c))
;;                                  (reduced true)
;;                                  result))
;;                              false indices)))
;;         play (fn [winner index]
;;                (let [[r c] (moves index)
;;                      chess (if (even? index) 1 2)]
;;                  (aset matrix r c chess)
;;                  (let [winner' (get-winner matrix)]
;;                    (if (nil? winner')
;;                      winner
;;                      (reduced winner')))))
;;         winner (reduce play nil (range (count moves)))]
;;     (cond
;;       (= winner 1) "A"
;;       (= winner 2) "B"
;;       (pending?) "Pending"
;;       :else "Draw")))
;;  (map tictactoe [[[0 0] [2 0] [1 1] [2 1] [2 2]] [[0 0] [1 1] [0 1] [0 2] [1 0] [2 0]]
;;                 [[0 0] [1 1] [2 0] [1 0] [1 2] [2 1] [0 1] [0 2] [2 2]]
;;                 [[0 0] [1 1]]])

;; ;;1281
;; (defn subtract-product-and-sum [n]
;;   (let [integer-digits (fn [n]
;;                         (map #(- (int %) (int \0)) (vec (str n))))
;;         sum (fn [xs] (apply + xs))
;;         product (fn [xs] (apply * xs))
;;         digits (integer-digits n)]
;;     (- (product digits) (sum digits))))
;; (map subtract-product-and-sum [234 4421])

;; ;;1287
;; (defn find-special-integer1 [arr]
;;   (let [freqs (frequencies arr)
;;         threshold (/ (count arr) 4)
;;         find-special (fn [result num]
;;                        (if (> (get freqs num) threshold)
;;                          (reduced num)
;;                          result))]
;;     (reduce find-special nil (keys freqs))))

;; (defn find-special-integer [arr]
;;   (let [len (count arr)
;;         delta (quot len 4)
;;         find-special (fn [result index]
;;           (if (= (arr index) (arr (+ index delta)))
;;             (reduced (arr index))
;;             result))]
;;     (reduce find-special nil (range (- len delta)))))
;; (map find-special-integer [[1 2 2 6 6 6 6 7 10] [1 1]])

;; ;;1295
;; (defn find-numbers [nums]
;;   (->> (map str nums)
;;        (map count)
;;        (filter even?)
;;        (count)))
;; (map find-numbers [[12 345 2 6 7896] [555 901 482 1771]])

;; ;;1299
;; (defn replace-elements [arr]
;;   (let [replace-element (fn [[results max-num] index]
;;                           (let [max-num' (max (arr index) max-num)]
;;                             [(cons max-num results) max-num']))
;;         indices (reverse (range (dec (count arr))))]
;;     (reduce replace-element [[-1] (last arr)] indices)))
;; (map replace-elements [[17 18 5 4 6 1] [400]])

;; ;;1304
;; (defn sum-zero [n]
;;   (let [initial (if (odd? n) [0] [])
;;         append-sum-zero-pair (fn [result i]
;;                                (conj result i (- 0 i)))]
;;     (reduce append-sum-zero-pair initial (range 1 (inc (quot n 2))))))
;; (map sum-zero [5 3 1])

;; ;;1309
;; (defn freq-alphabets [s]
;;   (let [index->entry (fn [i]
;;                        [(if (< i 10) (str i) (str i "#"))
;;                         (str (char (+ (int \a) (dec i))))])
;;         dict (into {} (map index->entry (range 1 27)))]
;;     (letfn [(single-digit? [s index]
;;               (str/includes? "123456789" (subs s index (inc index))))
;;             (decode-single-digit [s index]
;;               (get dict (subs s index (inc index))))
;;             (decode-multiple-digits [s index]
;;               (get dict (subs s (- index 2) (inc index))))
;;             (decode [s index]
;;               (cond
;;                 (neg? index) ""
;;                 (and (>= index 0) (single-digit? s index)) (str (decode s (dec index)) (decode-single-digit s index))
;;                 :else (str (decode s (- index 3)) (decode-multiple-digits s index))))]
;;       (decode s (dec (count s))))))

;; (defn freq-alphabets [s]
;;   (let [decode-double-digits #(str (char (+ (int (Integer/parseInt (subs %1 0 2))) (dec (int \a)))))
;;         decode-single-digit (fn [digit] (str (char (+ (int (Integer/parseInt (subs digit 0 1))) (dec (int \a))))))]
;;   (->> (str/replace s #"\d\d#" decode-double-digits)
;;        (#(str/replace % #"\d" decode-single-digit)))))
;; (map freq-alphabets ["10#11#12" "1326#" "25#" "12345678910#11#12#13#14#15#16#17#18#19#20#21#22#23#24#25#26#"])

;; ;;1313
;; (defn decompress-RLE-list [nums]
;;   (let [decompress (fn [index]
;;                      (let [freq (nums index)
;;                            val (nums (inc index))]
;;                        (take freq (cycle [val]))))
;;         indices (range 0 (count nums) 2)]
;;     (flatten (map decompress indices))))
;; (map decompress-RLE-list [[1 2 3 4] [1 1 2 3]])

;; ;;1317
;; (defn get-no-zero-integers [n]
;;   (let [non-zero? (fn [num]
;;                     (= (.indexOf (str num (- n num)) "0") -1))
;;         check-non-zero-integers (fn [result num]
;;                                   (if (non-zero? num)
;;                                     (reduced [num (- n num)])
;;                                     result))]
;;     (reduce check-non-zero-integers [] (range 1 n))))
;; (map get-no-zero-integers [2 11 10000 69 1010])

;; ;; ;;1323
;; ;; (defn maximum-69-number1 [num]
;; ;;   (let [s (str num)
;; ;;         index (.indexOf s "6")
;; ;;         replace-first-6 (fn [s]
;; ;;                           (Integer/parseInt (str (subs s 0 index) "9" (subs s (inc index)))))]
;; ;;     (if (= index -1)
;; ;;       num
;; ;;       (replace-first-6 s))))

;; ;; (defn maximum-69-number [num]
;; ;;   (->> (str num)
;; ;;        (#(str/replace-first % "6" "9"))
;; ;;        (Integer/parseInt)))
;; ;; (map maximum-69-number [9669 9996 9999])

;; ;;1331
;; (defn array-rank-transform [arr]
;;   (let [nums (sort (distinct arr))
;;         rank-map (into {} (map vector nums (range 1 (inc (count nums)))))]
;;     (map #(get rank-map %) arr)))
;; (map array-rank-transform [[40 10 20 30] [100 100 100] [37 12 28 9 100 56 80 5 12]])

;; ;;1332
;; (defn remove-palindrome-sub [s]
;;   (if (= s (str/reverse s))
;;     1
;;     2))
;; (map remove-palindrome-sub ["ababa" "abb" "baabb"])

;; ;;1337
;; (defn k-weakest-rows [mat k]
;;   (let [sum (fn [r] (apply + (mat r)))
;;         compare-row (fn [l r]
;;                       (if (= (sum l) (sum r))
;;                         (compare l r)
;;                         (compare (sum l) (sum r))))]
;;   (take k (sort compare-row (range (count mat))))))
;; (map (partial apply k-weakest-rows) ['([[1 1 0 0 0]
;;                                          [1 1 1 1 0]
;;                                          [1 0 0 0 0]
;;                                          [1 1 0 0 0]
;;                                          [1 1 1 1 1]] 3)
;;                                       '([[1 0 0 0]
;;                                          [1 1 1 1]
;;                                          [1 0 0 0]
;;                                          [1 0 0 0]] 2)])

;; ;;1342
;; (defn number-of-steps [num]
;;   (letfn [(get-steps [num]
;;             (cond
;;               (zero? num) 0
;;               (even? num) (inc (get-steps (quot num 2)))
;;               :else (inc (get-steps (dec num)))))]
;;     (get-steps num)))
;; (map number-of-steps [14 8 123])

;; ;;1346
;; (defn check-if-exists1 [arr]
;;   (let [freqs (frequencies arr)
;;         nums (sort (keys freqs))
;;         multiple-zeros (> (or (get freqs 0) 0) 1)
;;         check (fn [result num]
;;                 (if (not (nil? (get freqs (* 2 num))))
;;                   (reduced true)
;;                   result))]
;;     (if multiple-zeros
;;      true
;;       (reduce check false nums))))

;; (defn check-if-exists [arr]
;;   (let [exists? (fn [num-set num] (or (contains? num-set (* 2 num))
;;                     (and (zero? (rem num 2)) (contains? num-set (quot num 2)))))
;;         check (fn [[result num-set] num]
;;                 (if (exists? num-set num)
;;                  (reduced [true num-set])
;;                   [result (conj num-set num)]))]
;;     (first (reduce check [false #{}] arr))))
;; (map check-if-exists [[10 2 5 3] [7 1 14 11] [3 1 7 11]])

;; ;;1351
;; ;; (defn count-negatives [grid]
;; ;;   (let [row (count grid)
;; ;;         col (count (grid 0))
;; ;;         count-negative (fn [r sum c]
;; ;;                          (if (neg? ((grid r) c))
;; ;;                            (inc sum)
;; ;;                            (reduced sum)))
;; ;;         count-negs (fn [r]
;; ;;                      (let [cols  (reverse (range col))]
;; ;;                      (reduce #(count-negative r %1 %2) 0 cols)))]
;; ;;     (apply + (map count-negs (range row)))))
;; ;; (map count-negatives [[[4 3 2 -1] [3 2 1 -1] [1 1 -1 -2] [-1 -1 -2 -3]] [[3 2] [1 0]] [[1 -1] [-1 -1]] [[-1]]])

;; ;;1356
;; (defn sort-by-bits [arr]
;;   (let [integer-digits (fn [n]
;;                          (Integer/toString n 2)
;;                          )
;;         count-1-bits (fn [n]
;;                        (->> (integer-digits n)
;;                             (vec)
;;                             (filter #(= % \1))
;;                             (count)))
;;         compare-bits (fn [l r]
;;                        (let [bits1 (count-1-bits l)
;;                              bits2 (count-1-bits r)]
;;                          (if (= bits1 bits2)
;;                            (compare l r)
;;                            (compare bits1 bits2))))]
;;     (sort compare-bits arr)))
;; (map sort-by-bits [[0 1 2 3 4 5 6 7 8] [1024 512 256 128 64 32 16 8 4 2 1] [10000 10000] [2 3 5 7 11 13 17 19] [10 100 1000 10000]])

;; ;;1360
;; (defn days-between-date [date1 date2]
;;   (let [[date1 date2] (if (neg? (compare date1 date2))
;;                         [date1 date2]
;;                         [date2 date1])
;;         to-date-components (fn [date]
;;                              (->> (str/split date #"-")
;;                                   (map #(Integer/parseInt %))
;;                                   (vec)))
;;         month-days [31 28 31 30 31 30 31 31 30 31 30 31]
;;         divisible? (fn [a b] (zero? (rem a b)))
;;         leap-year? (fn [year] (or (and (divisible? year 4) (divisible? year 100))
;;                                   (divisible? year 400)))
;;         leap-day (fn [year month]
;;                    (if (and (leap-year? year) (> month 2))
;;                      1
;;                      0))
;;         days-since-1971 (fn [date]
;;                           (let [[year month day] (to-date-components date)
;;                                 part1 (apply + (map #(if (leap-year? %) 366 365) (range 1971 year)))
;;                                 part2 (+ (apply + (map #(month-days (dec %)) (range 1 month)))
;;                                          (leap-day year month))
;;                                 part3 day]
;;                            (+ part1 part2 part3)))
;;         date-difference (fn [date1 date2]
;;                           (- (days-since-1971 date2)
;;                              (days-since-1971 date1)))]
;;     (date-difference date1 date2)))
;; (map (partial apply days-between-date) ['("2019-06-29" "2019-06-30") '("2020-01-15" "2019-12-31")])

;; ;;1365
;; (defn smaller-numbers-than-current [nums]
;;   (let [sorted-nums (sort nums)
;;         pairs (map vector (range (count sorted-nums)) sorted-nums)
;;         build-index (fn [m [index num]]
;;                       (if (nil? (get m num))
;;                         (assoc m num index)
;;                         m))
;;         index-map (reduce build-index {} pairs)]
;;     (map (fn [num] (get index-map num)) nums)))
;; (map smaller-numbers-than-current [[8 1 2 2 3] [6 5 4 8] [7 7 7 7]])

;; ;;1370
;; (defn sort-string [s]
;;   (let [freqs (frequencies (vec s))]
;;     (letfn [(rearrange-string [freqs acsending]
;;               (let [chars (->> (str/join "" (sort (keys freqs)))
;;                                (#(if acsending % (str/reverse %))))
;;                     update-count (fn [m c]
;;                                    (let [cnt (get m c)]
;;                                      (if (> cnt 1)
;;                                        (assoc m c (dec cnt))
;;                                        (dissoc m c))))
;;                     freqs' (reduce update-count freqs chars)]
;;                 (if (empty? freqs)
;;                   ""
;;                   (str chars (rearrange-string freqs' (not acsending))))))]
;;       (rearrange-string freqs true))))
;; (map sort-string ["aaaabbbbcccc" "rat" "leetcode" "ggggggg" "spo"])

;; ;;1374
;; (defn generate-the-string [n]
;;   (let [generate-even-string (fn [n]
;;                                (str "x" (str/join "" (take (dec n) (cycle "y")))))]
;;     (cond
;;       (= n 1) "z"
;;       (even? n) (generate-even-string n)
;;       :else (str (generate-even-string (dec n)) "z"))))
;; (map generate-the-string [4 2 7])

;; ;;1380
;; (defn lucky-numbers [matrix]
;;   (let [row (count matrix)
;;         mins (mapv #(apply min %) matrix)
;;         ->num-col-pair (fn [r]
;;                          [(mins r) (.indexOf (matrix r) (mins r))])
;;         min-num-col-pairs (mapv ->num-col-pair (range row))
;;         lucky-number? (fn [[min-num c]]
;;                         (= min-num (reduce #(max %1 ((matrix %2) c)) min-num (range row))))]
;;     (->> (filter lucky-number? min-num-col-pairs)
;;          (mapv first))))
;; (map lucky-numbers [[[3 7 8] [9 11 13] [15 16 17]] [[1 10 4 2] [9 3 8 7] [15 16 17 12]] [[7 8] [1 2]] [[3 6] [7 1] [5 2] [4 8]]])

;; ;;1385
;; (defn find-the-distance-value [arr1 arr2 d]
;;   (let [abs (fn [n] (if (neg? n)
;;                       (- 0 n)
;;                       n))
;;         count-distance (fn [sum num]
;;                          (if (every? #(> (abs (- % num)) d) arr2)
;;                            (inc sum)
;;                            sum))]
;;    (reduce count-distance 0 arr1)))
;; (map (partial apply find-the-distance-value) ['([4 5 8] [10 9 1 8] 2) '([1 4 2 3] [-4 -3 6 10 20 30] 3) '([2 1 100 3] [-5 -2 10 -3 7] 6)])

;; ;;1389
;; (defn create-target-array [nums index]
;;   (let [insert (fn [result i]
;;                  (let [num (nums i)
;;                        index' (index i)
;;                        part1 (conj (subvec result 0 index') num)
;;                        part2 (subvec result index')]
;;                      (vec (concat part1 part2))))]
;;     (reduce insert [] (range (count nums)))))
;;  (map (partial apply create-target-array) ['([0 1 2 3 4] [0 1 2 2 1])
;;                                           '([1 2 3 4 0]  [0 1 2 3 0])
;;                                           '([1] [0])])
;; ;;1394
;; (defn find-lucky [arr]
;;   (->> (into [] (frequencies arr))
;;        (filter #(apply = %))
;;        (map first)
;;        (#(if (empty? %) -1 (apply max %)))))
;; (map find-lucky [[2,2,3,4] [1,2,2,3,3,3] [2,2,2,3,3] [5] [7,7,7,7,7,7,7]])

;; ;;1399
;; (defn count-largest-group [n]
;;   (let [sum-digits1 (fn [n]
;;                       (->> (vec (str n))
;;                            (map #(Integer/parseInt (str %)))
;;                            (apply +)))

;;         sum-digits (fn [n]
;;                      (loop [sum 0 num n]
;;                        (if (< num 10)
;;                          (+ sum num)
;;                          (let [sum' (+ sum (rem num 10))
;;                                num' (quot num 10)]
;;                            (recur sum' num')))))
;;         add-sum-num (fn [m index]
;;                       (let [sum (sum-digits index)
;;                             cnt (or (get m sum) 0)]
;;                         (assoc m sum (inc cnt))))
;;         counts (vals (reduce add-sum-num {} (range 1 (inc n))))
;;         max-size (apply max counts)]
;;     (get (frequencies counts) max-size)))
;; (map count-largest-group [13 2 15 24])

;; ;;1403
;; (defn min-subsequence [nums]
;;   (let [nums (sort > nums)
;;         sum (apply + nums)
;;         append-element (fn [[result s] num]
;;                          (let [sum' (+ s num)]
;;                            (if (> sum' (- sum sum'))
;;                              (reduced (conj result num))
;;                              [(conj result num) sum'])))]
;;     (reduce append-element [[] 0] nums)))
;; (map min-subsequence [[4 3 10 9 8] [4 4 7 6 7] [6]])

;; ;;1408
;; (defn string-matching1 [words]
;;   (let [words (vec (sort #(compare (count %1) (count %2)) words))
;;         len (count words)
;;         substring? (fn [index]
;;                      (let [needle (words index)
;;                            check-substring (fn [result haystack]
;;                                              (if (str/includes? haystack needle)
;;                                                (reduced true)
;;                                                result))
;;                            haystacks (subvec words (inc index))]
;;                        (reduce check-substring false haystacks)))]
;;     (->> (filter substring? (range (dec len)))
;;          (flatten)
;;          (map #(words %)))))
;; (defn string-matching [words]
;;   (let [words (sort #(compare (count %1) (count %2)) words)
;;         sentence (str/join " " words)
;;         count-substrings (fn [needle haystack] (count (re-seq (re-pattern needle) haystack)))
;;         substring? #(> (count-substrings % sentence) 1)]
;;     (filter substring? (drop-last words))))
;; (map string-matching [["mass" "as" "hero" "superhero"] ["leetcode" "et" "code"] ["blue" "green" "bu"]])


;; ;; ;;1413
;; (defn min-start-value [nums]
;;   (let [len (count nums)
;;         index (reduce (fn [result index]
;;                         (if (neg? (nums index))
;;                           (reduced index)
;;                           result))
;;                       0 (reverse (range len)))
;;         check-min-sum (fn [[min-sum sum] num]
;;                         [(min min-sum (+ sum num)) (+ sum num)])
;;         initial-values  [(first nums) (first nums)]
;;         get-min-sum (fn [nums]
;;                       (first (reduce check-min-sum initial-values (subvec nums 1 (inc index)))))]
;;     (max 1 (- 1 (get-min-sum nums)))))
;; (map min-start-value [[-3 2 -3 4 2] [1 2] [1 -2 -3]])

;; ;;1417
;; (defn reformat [s]
;;   (let [cs (vec s)
;;         letters (vec (filter #(Character/isLetter %) cs))
;;         digits (vec (filter #(Character/isDigit %) cs))
;;         abs (fn [n] (if (neg? n) (- 0 n) n))
;;         valid-string (>= 1 (abs (- (count letters) (count digits))))
;;         [long-list short-list] (if (> (count letters) (count digits)) [letters digits] [digits letters])
;;         ;; reformat' (fn [short-list long-list]
;;         ;;             (->> (vec (flatten (map vector long-list short-list)))
;;         ;;               (#(concat % (subvec long-list (count short-list))))
;;         ;;               (str/join "")))]
;;         reformat' (fn [long-list short-list]
;;                     (let [add-char (fn [[result index1 index2] index]
;;                                      (if (even? index)
;;                                        [(conj result (long-list index1)) (inc index1) index2]
;;                                        [(conj result (short-list index2)) index1 (inc index2)]))
;;                           indices (range (count s))]
;;                       (->> (reduce add-char [[] 0 0] indices)
;;                            (first)
;;                            (flatten)
;;                            (str/join "")
;;                            )))]
;;     (if valid-string
;;       (reformat' long-list short-list)
;;       "")))
;; (map reformat ["a0b1c2" "leetcode" "1229857369" "covid2019" "ab123"])

;; ;;1422
;; (defn max-score [s]
;;   (let [bits (vec s)
;;         num-of-ones (count (filter #(= \1 %) bits))
;;         get-max-score (fn [[max-score zeros ones] c]
;;           (let [zeros' (+ zeros (if (= c \0) 1 0))
;;                 ones' (- ones (if (= c \0) 0 1))]
;;             [(max max-score (+ zeros' ones')) zeros' ones']))]
;;    (first (reduce get-max-score [0 0 num-of-ones] (drop-last bits)))))
;; (map max-score ["011101" "00111" "1111"])

;; ;;1426
;; (defn count-elements [arr]
;;   (let [freqs (frequencies arr)
;;         nums (sort (keys freqs))
;;         count-elements (fn [s num]
;;                          (let [cnt (get freqs num)
;;                                not-exist (nil? (get freqs (inc num)))]
;;                            (if not-exist
;;                             s
;;                              (+ s cnt))))]

;;     (reduce count-elements 0 (drop-last nums))))
;; (map count-elements [[1 2 3] [1 1 3 3 5 5 7 7] [1 3 2 3 5 0] [1 1 2 2] [1 1 2]])


;; ;;1427  GOOD
;; (defn string-shift [s shift]
;;   (let [len (count s)
;;         simplify-shift (fn [shift]
;;                          (let [add-amount (fn [[amount0 amount1] [index amount]]
;;                                             (if (zero? index)
;;                                               [(+ amount0 amount) amount1]
;;                                               [amount0 (+ amount1 amount)]))
;;                                [amount0 amount1] (reduce add-amount [0 0] shift)]
;;                            (if (< amount0 amount1)
;;                              [1 (- amount1 amount0)]
;;                              [0 (- amount0 amount1)])))
;;         [index amount] (simplify-shift shift)]
;;     (if (zero? index)
;;       (str (subs s amount) (subs s 0 amount)))
;;       (str (subs s (- len amount)) (subs s 0 (- len amount)))))
;; (map (partial apply string-shift) ['("abc" [[0,1],[1,2]]) '("abcdefg" [[1,1],[1,1],[0,2],[1,3]])])

;; ;;1431
;; (defn kids-with-candies [candies extra-candies]
;;   (let [max-amount (apply max candies)
;;         max-candies? (fn [c] (>= (+ c extra-candies) max-amount))]
;;     (map max-candies? candies)))
;; (map (partial apply kids-with-candies) ['([2 3 5 1 3] 3) '([4 2 1 1 2] 1) '([12 1 12] 10)])

;; ;;1436
;; (defn dest-city [paths]
;;   (let [outgoing-cities (map first paths)
;;         all-cities (flatten paths)]
;;     (vec (set/difference (set all-cities) (set outgoing-cities)))))
;; (map dest-city [[["London" "New York"] ["New York" "Lima"] ["Lima" "Sao Paulo"]]
;;                 [["B" "C"] ["D" "B"] ["C" "A"]]
;;                 [["A" "Z"]]])
;; ;;1437
;; (defn k-length-apart [nums k]
;;   (let [indices (vec (filter #(= (nums %) 1) (range (count nums))))
;;         distances (map #(dec (- (indices %) (indices (dec %)))) (range 1 (count indices)))]
;;     (every? #(>= % k) distances)))
;; (map (partial apply k-length-apart) ['([1 0 0 0 1 0 0 1] 2) '([1 0 0 1 0 1] 2) '([1 1 1 1 1] 0) '([0 1 0 1] 1)])

;; ;;1441
;; (defn build-array [target n]
;;   (let [start 1
;;         end (last target)
;;         nums (range start (inc end))
;;         target-set (set target)
;;         ->operations #(if (contains? target-set %) ["Push"]
;;                         ["Push" "Pop"])]
;;     (flatten (map ->operations nums))))
;; (map (partial apply build-array) ['([1 3] 3) '([1 2 3] 3) '([1 2] 4) '([2 3 4] 4)])

;; ;;1446
;; (defn max-power [s]
;;   (let [cs (vec s)
;;         get-max-power (fn [[max-value len] index]
;;                         (cond
;;                           (= index (count cs)) [(max max-value len) 0]
;;                           (= (cs (dec index)) (cs index)) [(max max-value (inc len)) (inc len)]
;;                           :else [(max max-value len) 0]))
;;         indices (range 1 (inc (count s)))]
;;     (first (reduce get-max-power [0 1] indices))))
;; (map max-power ["leetcode" "abbcccddddeeeeedcba" "triplepillooooow" "hooraaaaaaaaaaay" "tourist"])

;; ;;1450
;; (defn busy-student [start-time end-time query-time]
;;   (let [busy-time (map vector start-time end-time)
;;         busy? (fn [[start end]]
;;                 (and (<= start query-time) (<= query-time end)))]
;;     (count (filter busy? busy-time))))
;; (map (partial apply busy-student) ['([4] [4] 4) '([4] [4] 5) '([1 1 1 1] [1 3 2 4] 7) '([9 8 7 6 5 4 3 2 1] [10 10 10 10 10 10 10 10 10] 5)])

;; ;;1455
;; (defn is-prefix-of-words [sentence search-word]
;;   (let [words (vec (re-seq #"\w+" sentence))
;;         find-index (fn [result index]
;;                      (if (str/starts-with? (words index) search-word)
;;                        (reduced (inc index))
;;                        result))]
;;     (reduce find-index -1 (range (count words)))))
;; (map (partial apply is-prefix-of-words) ['("i love eating burger" "burg")
;;                                          '("this problem is an easy problem" "pro")
;;                                          '("i am tired" "you")
;;                                          '("i use triple pillow" "pill")
;;                                          '("hello from the other side" "they")])
;; ;;1460
;; (defn can-be-equal [target arr]
;;  (apply = (map sort [target arr])))
;; ;  (apply = (map frequencies [target arr])))
;; (map (partial apply can-be-equal) ['([1 2 3 4] [2 4 1 3]) '([7] [7]) '([3 7 9] [3 7 11]) '([1 1 1 1 1] [1 1 1 1 1])])

;; ;;1464
;; (defn max-product [nums]
;;   (let [values (vec (take 2 (sort > nums)))]
;;     (apply * (map dec values))))
;; (map max-product [[3 4 5 2] [1 5 4 5] [3 7]])

;; ;;1470
;; (defn shuffle [nums n]
;;   (flatten (map vector (subvec nums 0 n) (subvec nums n))))
;; (map (partial apply shuffle) ['([2 5 1 3 4 7] 3) '([1 2 3 4 4 3 2 1] 4) '([1 1 2 2] 2)])

;; ;;1475
;; (defn final-prices1 [prices]
;;   (let [get-discount' (fn [price result discount]
;;                         (if (<= discount price)
;;                           (reduced discount)
;;                           result))
;;         get-discount (fn [index]
;;                        (let [price (prices index)
;;                              indices (subvec prices (inc index))]
;;                          (reduce (partial get-discount' price) 0 indices)))
;;         get-final-price (fn [index]
;;                           (let [price (prices index)
;;                                 discount (get-discount index)]
;;                             (- price discount)))]
;;     (map get-final-price (range (count prices)))))
;; (defn final-prices [prices]
;;   (let [result (into-array prices)
;;         update-final-price (fn [result stack index]
;;                              (reduce (fn [stack _]
;;                                        (if (and (not-empty stack) (>= (prices (last stack)) (prices index)))
;;                                          (doseq [index' [(last stack)]]
;;                                            (aset result index' (- (aget result index' (prices index)))
;;                                                  (drop-last stack)))
;;                                          (reduced stack))) [] (range (count stack))))
;;         get-final-price (fn [result stack index]
;;                           (let [stack' (update-final-price result stack index)]
;;                             (println "a:")
;;                             (println stack)
;;                             (println "b:")
;;                             (println stack')
;;                             (conj stack' index)))]
;;     (force (reduce (partial get-final-price result) [] (range (count prices))))
;;     (vec result)))
;; (map final-prices [[8 4 6 2 3] [1 2 3 4 5] [10 1 1 6]])

;;1480
;; (defn running-sum [nums]
;;   (reduce #(conj %1 (+ (last %1) %2)) [(first nums)] (rest nums)))
;; (map running-sum [[1 2 3 4] [1 1 1 1 1] [3 1 2 10 1]])

;; ;;1486
;; (defn xor-operation [n start]
;;   (let [nums (map (fn [i] (+ start (* 2 i))) (range n))]
;;        (reduce #(bit-xor %2 %1) (first nums) (rest nums))))
;; (map (partial apply xor-operation) ['(5 0) '(4 3) '(1 7) '(10 5)])

;; ;;1491
;; (defn average [salary]
;;   (let [n (- (count salary) 2)
;;         sum (- (apply + salary) (apply max salary) (apply min salary))]
;;     (float (/ sum n))))
;; (map average [[4000 3000 1000 2000] [1000 2000 3000] [6000 5000 4000 3000 2000 1000] [8000 9000 2000 3000 6000 1000]])

;; ;;1496
;; (defn is-path-crossing [path]
;;   (let [move-vec-map {\N [0 1] \S [0 -1] \E [1 0] \W [-1 0]}
;;         to-2d-vector (fn [move]
;;                        (get move-vec-map move))
;;         vectors (mapv to-2d-vector (vec path))
;;         crossing? (fn [[vx vy] [u1 u2]]
;;           (let [vx' (+ vx u1)
;;                 vy' (+ vy u2)]
;;             (if (and (zero? vx') (zero? vy'))
;;               (reduced [vx' vy'])
;;               [vx' vy'])))]
;;     (= [0 0] (reduce crossing? [0 0] vectors))))
;; (map is-path-crossing ["NES" "NESWW"])

;; ;;1502
;; (defn can-make-arithmetic-progression [arr]
;;   (let [arr (vec (sort arr))
;;         ->difference (fn [index]
;;                        (- (arr index) (arr (dec index))))
;;         indices (range 1 (count arr))]
;;    (= 1 (count (set (map ->difference indices))))))
;; (map can-make-arithmetic-progression [[3 5 1] [1 2 4]])

;; ;;1507
;; (defn reformat-date [date]
;;   (let [months {"Jan" "01"  "Feb" "02"  "Mar" "03"  "Apr" "04"  "May" "05"  "Jun" "06"  "Jul" "07"  "Aug" "08"  "Sep" "09"  "Oct" "10"  "Nov" "11"  "Dec" "12"}
;;         [day month year] (vec (str/split date #"\s"))
;;     components [year (get months month) (first (re-seq #"\d+" day))]]
;;     (str/join "-" components)))
;; (map reformat-date ["20th Oct 2052" "6th Jun 1933" "26th May 1960"])

;; ;;1512
;; (defn num-identical-pairs [nums]
;;   (let [counts (vals (frequencies nums))
;;         count-pairs #(quot (* % (dec %)) 2)]
;;     (apply + (map count-pairs counts))))
;; (map num-identical-pairs [[1,2,3,1,1,3] [1,1,1,1] [1,2,3]])

;; ;;1518
;; (defn num-water-bottles [num-bottles num-exchange]
;;   (letfn [(max-water-bottles [n empty-bottles]
;;             (let [empty-bottles' (+ empty-bottles n)
;;                   n' (quot empty-bottles' num-exchange)]
;;               (if (zero? n)
;;                 0
;;                 (+ n (max-water-bottles n' (rem empty-bottles' num-exchange))))))]
;;     (max-water-bottles num-bottles 0)))
;; (map (partial apply num-water-bottles) ['(9 3) '(15 4) '(5 5) '(2 3)])

;; ;;1523
;; (defn count-odds [low high]
;;   (let [delta (- high low)]
;;   (if (odd? delta) (inc (quot delta 2))
;;       (+ (quot delta 2) (if (odd? low) 1 0)))))
;; (map (partial apply count-odds) ['(3 7) '(8 10)])

;; ;;1528
;; (defn restore-string [s indices]
;;   (let [cs (vec s)
;;         result (make-array Character/TYPE (count s))]
;;     (doseq [i (range (count s))]
;;       (aset result (indices i) (cs i)))
;;     (str/join "" result)))
;; (map (partial apply restore-string) ['("codeleet" [4 5 6 7 0 2 1 3]) '("abc" [0 1 2]) '("aiohn" [3 1 4 2 0]) '("aaiougrt" [4 0 2 6 7 3 1 5]) '("art" [1 0 2])])

;; ;;1534
;; (defn count-good-triplets [arr a b c]
;;   (let [len (count arr)
;;         triplets (for [i (range (- len 2)) j (range (inc i) (- len 1)) k (range (inc j) len)]
;;                    [(arr i) (arr j) (arr k)])
;;         abs (fn [n]
;;               (if (neg? n) (- 0 n) n))
;;         good-triplet? (fn [[x1 x2 x3]]
;;                         (and (<= (abs (- x1 x2)) a)
;;                              (<= (abs (- x3 x2)) b)
;;                              (<= (abs (- x3 x1)) c)))]
;;     (count (filter good-triplet? triplets))))
;; (map (partial apply count-good-triplets) ['([3,0,1,1,9,7] 7 2 3) '([1,1,2,2,3] 0 0 1)])

;; ;;1539
;; (defn find-kth-positive1 [arr k]
;;   (let [positives (set arr)
;;         find-positive (fn [[result cnt] num]
;;                         (cond
;;                          (contains? positives num) [result cnt]
;;                           (= (inc cnt) k) (reduced [num (inc cnt)])
;;                           :else [result (inc cnt)]))
;;         nums (range 1 1001)]
;;    (first (reduce find-positive [nil 0] nums))))
;; (defn find-kth-positive [arr k]
;;   (letfn [(find-positive [left right]
;;             (let [middle (quot (+ left right) 2)]
;;               (cond
;;                 (>= left right) [left right]
;;                 (< (- (arr middle) (inc middle)) k) (find-positive (inc middle) right)
;;                 :else (find-positive left (dec middle)))))]
;;     (->> (find-positive 0 (dec (count arr)))
;;          (first)
;;          (#(+ % k))
;;          )))
;; (map (partial apply find-kth-positive) ['([2 3 4 7 11] 5) '([1 2 3 4] 2)])

;; ;;1544
;; (defn make-good [s]
;;   (let [len (count s)
;;         cs (vec s)
;;         indices (range len)
;;         bad-string?' (fn [result index]
;;                        (let [c1 (last result)
;;                              c2 (cs index)]
;;                          (and (= (Character/toLowerCase c1) (Character/toLowerCase c2))
;;                               (not= c1 c2))))

;;         bad-string? (fn [result index]
;;                       (if (empty? result) false
;;                           (bad-string?' result index)))
;;         remove-bad-string (fn [result index]
;;                             (if (bad-string? result index)
;;                               (vec (drop-last result))
;;                               (conj result (cs index))))]
;;     (str/join "" (reduce remove-bad-string [] indices))))
;; (map make-good ["leEeetcode" "abBAcC" "s"])

;; ;;1550
;; (defn three-consective-odds [arr]
;;   (let [len (count arr)]
;;     (loop [index 0 cnt 0]
;;       (cond
;;         (>= index len) false
;;         (and (odd? (arr index)) (= (inc cnt) 3)) true
;;         :else (recur (inc index) (if (even? (arr index)) 0 (inc cnt))
;;       )))))
;; (map three-consective-odds [[2 6 4 1] [1 2 34 3 4 5 7 23 12]])

;; ;;1556
;; (defn thousand-separator [n]
;;   (let [digits (str/reverse (str n))
;;         len (count digits)
;;         ->part (fn [start] (subs digits start (min len (+ start 3))))]
;;     (->> (map ->part (range 0 len 3))
;;          (map str/reverse)
;;          (reverse)
;;          (str/join "."))))
;; (map thousand-separator [987 1234 123456789 0])

;; ;;1560
;; (defn most-visited1 [n rounds]
;;   (let [indices (range (dec (count rounds)))
;;         add-visits (fn [visits index]
;;                      (concat visits (range (rounds index) (rounds (inc index)))))
;;         freqs (->> (reduce add-visits [(last rounds)] indices)
;;                    (frequencies))
;;         max-count (apply max (vals freqs))]
;;     (->> (keys freqs)
;;          (filter #(= (get freqs %) max-count))
;;          (sort))))
;; (defn most-visited [n rounds]
;;   (let [start (first rounds)
;;         end (last rounds)]
;;     (if (<= start end)
;;       (range start (inc end))
;;       (sort (concat (range end (inc n)) (range 1 (inc start)))))))
;; (map (partial apply most-visited) ['(4 [1 3 1 2]) '(2 [2 1 2 1 2 1 2 1 2]) '(7 [1 3 5 7])])

;; ;;1566
;; (defn contains-pattern1 [arr m k]
;;   (let [len (count arr)
;;         indices (range (- (inc len) (* m k)))
;;         find-pattern (fn [result index]
;;                        (let [indices (range index (+ index (* m k)) m)
;;                              parts (map #(subvec arr % (+ % m)) indices)]
;;                          (if (apply = parts)
;;                            (reduced true)
;;                            result)))]
;;     (reduce find-pattern false indices)))
;; (defn contains-pattern [arr m k]
;;   (let [len (count arr)
;;         indices (range (- len m))
;;         count-pattern (fn [[result cnt] index]
;;                         (if (= (arr index) (arr (+ index m)))
;;                           (if (= (inc cnt) (* (dec k) m))
;;                             (reduced [true (inc cnt)])
;;                             [result (inc cnt)])
;;                           [result 0]))]
;;    (first (reduce count-pattern [false 0] indices))))
;; (map (partial apply contains-pattern) ['([1 2 4 4 4 4] 1 3) '([1 2 1 2 1 1 1 3] 2 2) '([1 2 1 2 1 3] 2 3) '([1 2 3 1 2] 2 2) '([2 2 2 2] 2 3)])

;; ;;1572
;; (defn diagonal-sum [mat]
;;   (let [len (count mat)
;;         primary-indices (for [i (range len)] [i i])
;;         secondary-indices (for [i (range len)] [i (- (dec len) i)])
;;         indices (set (concat primary-indices secondary-indices))
;;         get-element (fn [[r c]]
;;                       ((mat r) c))]
;;     (apply + (map get-element indices))))
;; (map diagonal-sum [[[1 2 3]
;;                     [4 5 6]
;;                     [7 8 9]]
;;                    [[1 1 1 1]
;;                     [1 1 1 1]
;;                     [1 1 1 1]
;;                     [1 1 1 1]]
;;                    [[5]]])

;; ;;1576
;; (defn modify-string [s]
;;   (let [alphabet (set (map #(char (+ (int \a) %)) (range 26)))
;;         chars (vec (set/difference alphabet (set (vec (str/replace s #"\?" "")))))
;;         len (count chars)
;;         replace-question-mark (fn [[result index] c]
;;                                 (if (= c \?)
;;                                   [(conj result (chars index)) (rem (inc index) len)]
;;                                   [(conj result c) index]))]
;;     (->> (reduce replace-question-mark [[] 0] (vec s))
;;          (first)
;;          (str/join ""))))
;; (map modify-string ["?zs" "ubv?w" "j?qg??b" "??yw?ipkj?"])

;; ;;1582
;; (defn num-special [mat]
;;   (let [row (count mat)
;;         rows (filter #(= 1 (apply + (mat %))) (range row))
;;         cols (map #(.indexOf (mat %1) 1) rows)
;;         indices (map vector rows cols)
;;         special?' (fn [[r c] r']
;;                     (if (= r r')
;;                      (= 1 ((mat r') c))
;;                      (zero? ((mat r') c))))
;;         special? (fn [[r c]]
;;                    (every? #(special?' [r c] %) (range row)))]
;;     (count (filter special? indices))))
;; (map num-special [[[1 0 0]
;;                    [0 0 1]
;;                    [1 0 0]]
;;                   [[1 0 0]
;;                    [0 1 0]
;;                    [0 0 1]]
;;                   [[0 0 0 1]
;;                    [1 0 0 0]
;;                    [0 1 1 0]
;;                    [0 0 0 0]]
;;                   [[0 0 0 0 0]
;;                    [1 0 0 0 0]
;;                    [0 1 0 0 0]
;;                    [0 0 1 0 0]
;;                    [0 0 0 1 1]]])

;; ;;1588
;; (defn sum-odd-length-subarrays1 [arr]
;;   (let [lens  (range 1 (inc (count arr)) 2)
;;         sum-subarrays (fn [len]
;;                         (let [indexes (range 1 (- (count arr) (dec len)))
;;                               initial-sums [(apply + (take len arr))]
;;                               sum-subarray (fn [sums index]
;;                                              (let [sum' (- (last sums) (arr (dec index)))
;;                                                    num (arr (+ index (dec len)))
;;                                                    sum (+ sum' num)]
;;                                                (conj sums sum)))]
;;                          (apply + (reduce sum-subarray initial-sums indexes))))]
;;     (apply + (map sum-subarrays lens))))

;; (defn sum-odd-length-subarrays [arr]
;;   (let [n (count arr)
;;         count-frequncy (fn [n index]
;;                          (let [start (- n index)
;;                                end (inc index)]
;;                            (quot (inc (* start end)) 2)))]
;;    (apply + (map #(* (count-frequncy n %) (arr %)) (range n)))))
;; (map sum-odd-length-subarrays [[1 4 2 5 3] [1 2] [10 11 12]])

;; ;;1592
;; (defn reorder-spaces [text]
;;   (let [cs (vec text)
;;         num-spaces (count (filter #(Character/isSpace %) cs))
;;         words (re-seq #"\w+" (str/trim text))
;;         num-words (count words)
;;         spaces (fn [n] (str/join "" (take n (cycle [" "]))))
;;         [equal-spaces extra-spaces] (if (zero? (dec num-words))
;;                                       ["" (spaces num-spaces)]
;;                                       [(spaces (quot num-spaces (dec num-words))) (spaces (rem num-spaces (dec num-words)))])]
;;     (str (str/join equal-spaces words) extra-spaces)
;;     ))
;; (map reorder-spaces ["  this   is  a sentence " " practice   makes   perfect" "hello   world" "  walks  udp package   into  bar a"])

;; ;;1598
;; (defn min-operations [logs]
;;   (let [count-path (fn [sum path]
;;                       (cond
;;                         (= "./" path) sum
;;                         (= "../" path) (max 0 (dec sum))
;;                         :else (inc sum)))]
;;  (reduce count-path 0 logs)))
;; (map min-operations [["d1/" "d2/" "../" "d21/" "./"] ["d1/" "d2/" "./" "d3/" "../" "d31/"] ["d1/" "../" "../" "../"]])

;; ;;1608
;; (defn special-array1 [nums]
;;   (let [nums (vec (sort nums))
;;         len (count nums)
;;         find-x (fn [result x]
;;                           (cond
;;                             (and (= x len) (<= x (nums (- len x)))) (reduced x)
;;                             (and (<= x (nums (- len x))) (< (nums (dec (- len x))) x)) (reduced x)
;;                             :else result))
;;         xs (reverse (range 1 (inc len)))]
;;     (reduce find-x -1 xs)))
;; (defn special-array [nums]
;;   (let [nums (vec (sort > nums))
;;         len (count nums)
;;         find-x (fn [result x]
;;                  (cond
;;                    (and (= x len) (>= (nums (dec x)) x)) (reduced x)
;;                    (and (< x len) (>= (nums (dec x)) x) (> x (nums x))) (reduced x)
;;                    :else result))
;;         xs (range 1 (inc len))]
;;     (reduce find-x -1 xs)))
;; (map special-array [[3 5] [0 0] [0 4 3 0 4] [3 6 7 7 0]])

;; ;;1614
;; (defn max-depth [s]
;;   (let [parentheses (re-seq #"[\(\)]" s)
;;         get-max-depth (fn [[max-length lefts] c]
;;                         (if (= "(" c)
;;                           [(max max-length (inc lefts)) (inc lefts)]
;;                           [max-length (dec lefts)]))]
;;   (first (reduce get-max-depth [0 0] parentheses))))
;; (map max-depth ["(1+(2*3)+((8)/4))+1" "(1)+((2))+(((3)))" "1+(2*3)/(2-1)" "1"])

;; ;;1619
;; (defn trim-mean [arr]
;;   (let [arr (sort arr)
;;         len (count arr)
;;         len-of-smallest (quot len 20)
;;         len-of-largest len-of-smallest
;;     nums (drop-last len-of-largest (drop len-of-smallest arr))]
;;     (float (/ (apply + nums) (count nums)))))
;; (map trim-mean [[1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3] [6,2,7,5,1,2,0,3,10,2,5,0,5,5,0,8,7,6,8,0] [6,0,7,0,7,5,7,8,3,4,0,7,8,1,6,8,1,1,2,4,8,1,9,5,4,3,8,5,10,8,6,6,1,0,6,10,8,2,3,4] [9,7,8,7,7,8,4,4,6,8,8,7,6,8,8,9,2,6,0,0,1,10,8,6,3,3,5,1,10,9,0,7,10,0,10,4,1,10,6,9,3,6,0,0,2,7,0,6,7,2,9,7,7,3,0,1,6,1,10,3] [4,8,4,10,0,7,1,3,7,8,8,3,4,1,6,2,1,1,8,0,9,8,0,3,9,10,3,10,1,10,7,3,2,1,4,9,10,7,6,4,0,8,5,1,2,1,6,2,5,0,7,10,9,10,3,7,10,5,8,5,7,6,7,6,10,9,5,10,5,5,7,2,10,7,7,8,2,0,1,1]])

;; ;;1624
;; (defn max-length-between-equal-character [s]
;;   (let [cs (vec s)
;;         freqs (frequencies cs)
;;         multiple-chars (filter #(> (get freqs %) 1) (keys freqs))
;;         max-length-of-substring (fn [c]
;;                                   (let [left (.indexOf cs c)
;;                                         right (.lastIndexOf cs c)]
;;                                     (dec (- right left))))
;;         lengths (map max-length-of-substring multiple-chars)]
;;     (if (empty? lengths)
;;       -1
;;       (apply max lengths))))
;; (map max-length-between-equal-character ["aa" "abca" "cbzxy" "cabbac"])

;; ;;1629
;; (defn slowest-key [release-times key-pressed]
;;   (let [cs (vec key-pressed)
;;         append-duration (fn [result index]
;;                           (conj result (- (release-times index) (release-times (dec index)))))
;;         durations (reduce append-duration [(first release-times)] (range 1 (count release-times)))
;;         max-duration (apply max durations)
;;         add-key (fn [result index]
;;           (if (= max-duration (durations index))
;;             (conj result (cs index))
;;             result))]
;;     (->> (reduce add-key [] (range (count cs)))
;;          (sort)
;;          (last))))
;; (map (partial apply slowest-key) ['([9,29,49,50] "cbcd") '([12,23,36,46,62] "spuda")])

;; ;;1636
;; (defn frequency-sort1 [nums]
;;   (let [freqs (into [] (frequencies nums))
;;         compare-freq (fn [[num1 count1] [num2 count2]]
;;                        (if (= count1 count2)
;;                          (compare num2 num1)
;;                          (compare count1 count2)))
;;         num-map (->> (sort compare-freq freqs)
;;                      (map first)
;;                      (#(map vector % (range (count %))))
;;                      (into {}))
;;         compare-index (fn [l r]
;;           (compare (get num-map l) (get num-map r)))]
;;     (sort compare-index nums)))

;; (defn frequency-sort [nums]
;;   (let [freqs (frequencies nums)
;;         compare-freq (fn [l r]
;;                        (let [freq1 (get freqs l)
;;                              freq2 (get freqs r)]
;;                          (if (= freq1 freq2)
;;                            (compare r l)
;;                            (compare freq1 freq2))))]
;;     (sort compare-freq nums)))
;; (map frequency-sort [[1 1 2 2 2 3] [2 3 1 3 2] [-1 1 -6 4 5 -6 1 4 1]])

;; ;;1640
;; (defn can-form-array1 [arr pieces]
;;   (let [to-str (fn [nums]
;;                  (let [s (str nums)]
;;                    (subs s 1 (dec (count s)))))
;;         can-form? (fn [arr pieces]
;;                    (every? #(str/includes? (to-str arr) (to-str %)) pieces))]
;;   (if (not= (sort arr) (sort (flatten pieces)))
;;     false
;;     (can-form? arr pieces))))
;; (defn can-form-array [arr pieces]
;;   (let [->num-piece-pair (fn [piece] [(first piece) piece])
;;         piece-map (into {} (map ->num-piece-pair pieces))
;;         ->piece (fn [num] (or (get piece-map num) []))]
;;     (= (flatten (map ->piece arr)) arr)))
;; (map (partial apply can-form-array) ['([85] [[85]]) '([15 88] [[88] [15]]) '([49 18 16] [[16 18 49]]) '([91 4 64 78] [[78] [4 64] [91]]) '([1 3 5 7] [[2 4 6 8]])])

;; ;;1646
;; (defn get-maximum-generated [n]
;;   (let [generate-num  (fn [nums index]
;;                         (let [index' (quot index 2)
;;                               num (if (even? index)
;;                                     (nums index')
;;                                     (+ (nums index') (nums (inc index'))))]
;;                           (conj nums num)))
;;         generate-array (fn [n]
;;                          (reduce generate-num [0 1] (range 2 (inc n))))]
;;     (apply max (generate-array n))))
;; (map get-maximum-generated [7 2 3])

;;1652
;; (defn decrypt [code k]
;;   (let [len (count code)
;;         decode (fn [index]
;;                  (let [start (if (pos? k)
;;                                (inc index)
;;                                  (+ index k))
;;                        end (if (pos? k)
;;                              (+ index k)
;;                              (dec index))]
;;                  (cond
;;                    (zero? k) 0
;;                    (and (>= start 0) (< end len)) (apply + (subvec code start (inc end)))
;;                    (neg? start) (apply + (concat (subvec code (+ start len)) (subvec code 0 (inc end))))
;;                    :else (apply + (concat (subvec code start len) (subvec code 0 (rem (inc end) len)))))))]
;;     (map decode (range len))))
;; (map (partial apply decrypt) ['([5 7 1 4] 3) '([1 2 3 4] 0) '([2 4 9 3] -2)])

;;1662
;; (defn array-strings-are-equal [word1 word2]
;;   (let [len1 (count word1)
;;         len2 (count word2)
;;         equal-length (= (apply + (map count word1)) (apply + (map count word2)))
;;         same-string? (fn []
;;                        (loop [word1-index 0
;;                               index1 0
;;                               word2-index 0
;;                               index2 0]
;;                          (let [w1 #(word1 word1-index)
;;                                w2 #(word2 word2-index)]
;;                            (cond
;;                              (or (= word1-index len1) (= word2-index len2)) true
;;                              (>= index1 (count (w1))) (recur (inc word1-index) 0 word2-index index2)
;;                              (>= index2 (count (w2))) (recur word1-index index1 (inc word2-index) 0)
;;                              (not= (subs (w1) index1 (inc index1)) (subs (w2) index2 (inc index2))) false
;;                              :else (recur word1-index (inc index1) word2-index (inc index2))))))]
;;     (if equal-length
;;       (same-string?)
;;       false)))
;; (map (partial apply array-strings-are-equal) ['(["ab" "c"] ["a" "bc"]) '(["a" "cb"] ["ab" "c"]) '(["abc" "d" "defg"] ["abcddefg"])])

;; ;;1668
;; (defn max-repeating [s word]
;;   (letfn [(count-repeating [s]
;;             (let [index (.indexOf s word)]
;;               (if (= index -1)
;;                 0
;;                 (inc (count-repeating (subs s (+ index (count word)))))))
;;             )]
;;     (count-repeating s)))
;; (map (partial apply max-repeating) ['("ababc" "ab") '("ababc" "ba") '("ababc" "ac")])

;; ;;1672
;; (defn maximum-wealth [accounts]
;;   (apply max (map (partial apply +) accounts)))
;; (map maximum-wealth [[[1 2 3] [3 2 1]] [[1 5] [7 3] [3 5]] [[2 8 7] [7 1 3] [1 9 5]]])

;; ;;1678
;; (defn interpret [command]
;;   (->> (str/replace command #"\(al\)" "al")
;;        (#(str/replace % #"\(\)" "o"))))
;; (map interpret ["G()(al)" "G()()()()(al)" "(al)G(al)()()G"])

;; ;;1684
;; (defn count-consistent-strings [allowed words]
;;   (let [allowed-set (set (vec allowed))
;;         allowed? (fn [word]
;;                    (set/subset? (set (vec word)) allowed-set))]
;;     (count (filter allowed? words))))
;; (map (partial apply count-consistent-strings) ['("ab" ["ad" "bd" "aaab" "baa" "badab"])
;;                                                '("abc" ["a" "b" "c" "ab" "ac" "bc" "abc"])
;;                                                '("cad" ["cc" "acd" "b" "ba" "bac" "bad" "ac" "d"])])

;; ;;1688
;; (defn number-of-matches [n]
;;   (letfn [(count-matches [n]
;;             (let [matches (quot n 2)
;;                   n' (quot (inc n) 2)]
;;               (if (= n 1)
;;                 0
;;                 (+ matches (count-matches n')))))]
;;     (count-matches n)))
;; (map number-of-matches [7 14])

;; ;;1694
;; (defn reformat-number [number]
;;   (let [digits (str/join "" (re-seq #"\d" number))
;;         len (count digits)
;;         reformat-4digits (fn []
;;                            (let [s (subs digits (- len 4))]
;;                              (str "-" (subs s 0 2) "-" (subs s 2 4))))
;;         reformat-2digits (fn []
;;                            (if (< len 3)
;;                              (subs digits (- len 2))
;;                            (str "-" (subs digits (- len 2)))))
;;         reformat-3digits (fn [n]
;;                            (if (< len 3)
;;                              ""
;;                              (str/join "-" (map #(subs digits % (+ % 3)) (range 0 (- len n) 3)))))]

;;     (cond
;;       (= (rem len 3) 1) (str (reformat-3digits 4) (reformat-4digits))
;;       (= (rem len 3) 2) (str (reformat-3digits 2) (reformat-2digits))
;;       :else (reformat-3digits 0))))
;; (map reformat-number ["1-23-45 6" "123 4-567" "123 4-5678" "12" "--17-5 229 35-39475 "])

;; ;;1700
;; (defn count-students [students sandwiches]
;;   (letfn [(eat-lunch [students sandwiches]
;;             (let [eat (fn [[students' food] student]
;;                       (cond
;;                         (empty? food) (reduced [students' food])
;;                         (= student (first food)) [students' (rest food)]
;;                         :else [(conj students' student) food]))
;;                   [students' sandwiches']  (reduce eat [[] sandwiches] students)]
;;               (if (= sandwiches sandwiches')
;;                 (count sandwiches)
;;                 (eat-lunch students' sandwiches'))))]
;;     (eat-lunch students sandwiches)))
;; (map (partial apply count-students) ['([1 1 0 0] [0 1 0 1]) '([1 1 1 0 0 1] [1 0 0 0 1 1])])

;;1704
(defn halves-are-alike [s]
  (let [len (quot (count s) 2)
        count-vowels (fn [s]
                       (count (re-seq #"[aeiouAEIOU]" s)))]
    (apply = (map count-vowels [(subs s 0 len) (subs s len)]))))
(map halves-are-alike ["book" "textbook" "MerryChristmas" "AbCdEfGh"])

;;1708
(defn largest-subarray1 [nums k]
  (let [len (count nums)]
    (reduce (fn [max-array index]
              (let [new-array (subvec nums index (+ index k))
                    result (compare max-array new-array)]
                (case result
                  -1 new-array
                  1 max-array
                  new-array))) (subvec nums 0 k) (range 1 (- (inc len) k)))))

(defn largest-subarray [nums k]
  (let [max-value (apply max (subvec nums 0 (- (count nums) (dec k))))
        index (.indexOf nums max-value)]
    (subvec nums index (+ index k))))
(map (partial apply largest-subarray) ['([1 4 5 2 3] 3) '([1 4 5 2 3] 4) '([1 4 5 2 3] 1)])

;;1710
(defn maximum-units1 [box-types truck-size]
  (let [box-types (vec (sort (fn [l r] (compare (last r) (last l))) box-types))
        box-map' (into {} (map vector (range (count box-types)) box-types))]
    (loop [box-map box-map' index 0 num-boxes 0 num-units 0]
      (if (>= index (count box-types))
        num-units
        (let [[num units] (get box-map index)
              num-boxes' (+ num-boxes 1)]
              (cond
                (and (pos? num) (> num-boxes' truck-size)) num-units
                (= num 1) (recur (dissoc box-map index) (inc index) num-boxes' (+ num-units units))
                :else (recur (assoc box-map index [(dec num) units]) index num-boxes' (+ num-units units))))
    ))))

(defn maximum-units [box-types truck-size]
  (let [box-types (vec (sort (fn [l r] (compare (last r) (last l))) box-types))
        get-max-units (fn [[truck-size sum-of-units] index]
                        (let [[num units] (box-types index)
                              boxes (min truck-size num)]
                          (if (= (- truck-size boxes) 0)
                            (reduced [(- truck-size boxes) (+ sum-of-units (* boxes units))])
                            [(- truck-size boxes) (+ sum-of-units (* boxes units))])))
        indices (range (count box-types))]
    (last (reduce get-max-units [truck-size 0] indices))))
(map (partial apply maximum-units) ['([[1 3] [2 2] [3 1]] 4) '([[5 10] [2 5] [4 7] [3 9]] 10)])

;;1716
(defn total-money [n]
  (reduce (fn [sum index]
           (+ sum (+ (rem index 7) 1 (quot index 7))))
        0 (range n)))
(map total-money [4 10 20])

;;1720
(defn decode [encoded fst]
  (reduce (fn [nums e]
           (conj nums (bit-xor (last nums) e))) [fst] encoded))
(map (partial apply decode) ['([1 2 3] 1) '([6 2 7 3] 4)])

;;1725
(defn count-good-rectangles [rectangles]
  (let [widths (map #(apply min %) rectangles)
        max-len (apply max widths)]
   (count (filter #(= max-len %) widths))))
(map count-good-rectangles [[[5 8] [3 9] [5 12] [16 5]] [[2 3] [3 7] [4 3] [3 7]]])

;;1732
(defn largest-altitude [gain]
  (let [add-altitude (fn [altitudes net-gain]
                       (conj altitudes (+ (last altitudes) net-gain)))
       altitudes (reduce add-altitude [0] gain)]
    (apply max altitudes)))
(map largest-altitude [[-5 1 5 0 -7] [-4 -3 -2 -1 4 3 2]])

;;1736
(defn maximum-time [time]
  (let [[h1 h2 m1 m2] (map #(if (Character/isDigit %)
                              (- (int %) (int \0))
                              -1) (vec (str/replace time ":" "")))
        minutes [(if (neg? m1) 5 m1) (if (neg? m2) 9 m2)]
        hours (cond
                (or (= h1 0) (= h1 1))  [h1 (if (neg? h2) 9 h2)]
                (= h1 2) [h1 (if (neg? h2) 3 h2)]
                (and (neg? h1) (neg? h2)) [2 3]
                (and (neg? h1) (< h2 4)) [2 h2]
                :else [1 h2])]
    (str (hours 0) (hours 1) ":" (minutes 0) (minutes 1))))
(map maximum-time ["2?:?0" "0?:3?" "1?:22"])

;;1742
(defn count-balls [low-limit high-limit]
  (let [sum-digits (fn [n]
                     (->> (vec (str n))
                          (map str)
                          (map #(Integer/parseInt %))
                          (apply +)))
        balls (map sum-digits (range low-limit (inc high-limit)))]
    (apply max (vals (frequencies balls)))))
(map (partial apply count-balls) ['(1 10) '(5 15) '(19 28)])

;;1748
(defn sum-of-unique [nums]
  (->> (frequencies nums)
       (into [])
       (filter (fn [[num cnt]]
                 (= cnt 1)))
       (map first)
       (apply +)))
(map sum-of-unique [[1 2 3 2] [1 1 1 1 1] [1 2 3 4 5]])

;;1752
(defn check [nums]
  (let [haystack (vec (concat nums nums))
        needle (vec (sort nums))
        len (count needle)]
    (letfn [(check-if-sorted [start]
              (let [index (.indexOf (subvec haystack start) (first needle))
                    index' (+ start index)]
                (cond
                  (= index -1) false
                  (= (subvec haystack index' (min (count haystack) (+ index' len))) needle) true
                  :else (check-if-sorted (inc index')))))]
      (check-if-sorted 0))))
(map check [[3 4 5 1 2] [2 1 3 4] [1 2 3] [1 1 1] [2 1]])

;;1758
(defn min-operations [s]
  (let [cs (vec s)
        get-operations (fn [[counter0 counter1] index]
                         (let [bit (- (int (cs index)) (int \0))]
                           (cond
                             (and (even? index) (= bit 0)) [counter0 (inc counter1)]
                             (and (even? index) (= bit 1)) [(inc counter0) counter1]
                             (= bit 0) [(inc counter0) counter1]
                             :else [counter0 (inc counter1)])))
        indices (range (count s))]
    (->> (reduce get-operations [0 0] indices)
         (apply min))))
(map min-operations ["0100" "10" "1111"])

;;1763
(defn longest-nice-substring [s]
  (let [cs (vec s)
        freqs (frequencies cs)
        nice? (fn [c]
                (if (Character/isUpperCase c)
                  (not (nil? (get freqs (Character/toLowerCase c))))
                  (not (nil? (get freqs (Character/toUpperCase c))))))
        not-nice-chars (str/join "" (filter (comp nice?) (keys freqs)))
        pattern #(re-pattern (str "[" not-nice-chars "]+"))
        get-nice-substring (fn [s]
                             (let [nice-strs (vec (re-seq (pattern) s))
                                   max-len (apply max (map count nice-strs))]
                               (first (filter #(= (count %) max-len) nice-strs))))]
    (if (= not-nice-chars "")
      s
      (get-nice-substring s))))
(map longest-nice-substring ["YazaAay" "Bb" "c" "dDzeE"])

;;1768
(defn merge-alternately [word1 word2]
  (let [len1 (count word1)
        len2 (count word2)
        indices (range (max len1 len2))
        get-next-str (fn [word len index]
                       (if (< index len)
                         (subs word index (inc index))
                         ""))
        add-str (fn [result index]
                  (let [s1 (get-next-str word1 len1 index)
                        s2 (get-next-str word2 len2 index)]
                    (str result s1 s2)))]
    (reduce add-str "" indices)))
(map (partial apply merge-alternately) ['("abc" "pqr") '("ab" "pqrs") '("abcd" "pq")])

;;1773
(defn count-matches [items rule-key rule-value]
  (let [type-map {"type" 0 "color" 1 "name" 2}
        match-rule? (fn [item]
                      (let [value (item (get type-map rule-key))]
                        (= value rule-value)))]
   (count (filter match-rule? items))))
(map (partial apply count-matches) ['([["phone" "blue" "pixel"] ["computer" "silver" "lenovo"] ["phone" "gold" "iphone"]]
                                      "color"
                                      "silver")
                                    '([["phone" "blue" "pixel"] ["computer" "silver" "phone"] ["phone" "gold" "iphone"]]
                                      "type"
                                      "phone")])
;;1779
(defn nearest-valid-point [x y points]
  (let [valid? (fn [[x1 y1]]
                 (or (= x1 x) (= y1 y)))
        abs (fn [n] (if (neg? n) (- 0 n) n))
        get-distance (fn [[x1 y1]]
                       (+ (abs (- x1 x)) (abs (- y1 y))))
        valid-points (filter valid? points)
        distances (map get-distance valid-points)
        max-distance #(apply max distances)]
    (if (empty? distances)
      -1
     (first (filter #(= % (max-distance)) distances)))))
(map (partial apply nearest-valid-point)
     ['(3 4 [[1 2] [3 1] [2 4] [2 3] [4 4]])
      '(3 4 [[3 4]])
      '(3 4 [[2 3]])])

;;1784
(defn check-ones-segment [s]
  (->> (str/split s #"0+")
       (count)
       (#(< % 2))))
(map check-ones-segment ["1001" "110"])

;;1790
(defn are-almost-equal [s1 s2]
  (let [->char-pair (fn [index]
                      [(subs s1 index (inc index))
                       (subs s2 index (inc index))])
        compare-strings (fn [s1 s2]
                          (let [pairs (filter #(not= (first %) (last %)) (map ->char-pair (range (count s1))))]
                            (and (= 2 (count pairs)) (= (first pairs) (reverse (last pairs))))))]
    (cond
      (= s1 s2) true
      (not= (count s1) (count s2)) false
      :else (compare-strings s1 s2))))
(map (partial apply are-almost-equal) ['("bank" "kanb") '("attack" "defend") '("kelb" "kelb") '("abcd" "dcba")])

;;1791
(defn find-center [edges]
  (let [freqs (frequencies (flatten edges))
        n (inc (count edges))
        find-center-node (fn [result num]
          (if (= (dec n) (get freqs num))
            (reduced num)
            result))]
    (reduce find-center-node nil (range 1 (inc n)))))
(map find-center [[[1 2] [2 3] [4 2]] [[1 2] [5 1] [1 3] [1 4]]])

;;1796
(defn second-highest [s]
  (->> (set (re-seq #"\d" s))
       (mapv #(Integer/parseInt %))
       (sort >)
       (#(or (second %) -1))))
(map second-highest ["dfa12321afd" "abc1111"])

;;1800
;; (defn max-ascending-sum [nums]
;;   (let [find-max-ascending-sum (fn [[max-sum sum] index]
;;                                  (if (< (nums (dec index)) (nums index))
;;                                    [(max max-sum (+ sum (nums index))) (+ sum (nums index))]
;;                                    [max-sum (nums index)]))
;;         indexes (range 1 (count nums))]
;;     (->> (reduce find-max-ascending-sum [(first nums) (first nums)] indexes)
;;          (first))))
;; (map max-ascending-sum [[10 20 30 5 10 50] [10 20 30 40 50] [12 17 15 13 10 11 12] [100 10 1]])

;;1805
(defn num-different-integers [word]
  (let [append-digit (fn [s result index]
                       (let [c (subs s index (inc index))]
                         (if (and (= result "") (= c "0"))
                           result
                           (str result c))))
        remove-leading-zero (fn [s]
                              (let [indexes (range (count s))]
                                (reduce (partial append-digit s) "" indexes)))]
    (->> (re-seq #"\d+" word)
         (map remove-leading-zero)
         (set)
         (count))))
(map num-different-integers ["a123bc34d8ef34" "leet1234code234" "a1b01c001"])

;;1812
(defn square-is-white [coordinates]
  (let [[letter digit] (vec coordinates)
        x (inc (-(int letter) (int \a)))
        y (Integer/parseInt (str digit))]
    (= 1 (rem (+ x y) 2))))
(map square-is-white ["a1" "h3" "c7"])

;;1816
(defn truncate-sentence1 [s k]
  (->> (re-seq #"\w+" s)
       (take k)
       (str/join " ")
       ))
(defn truncate-sentence [s k]
  (let [len (count s)
        count-words (fn [[result words] index]
          (if (= " " (subs s index (inc index)))
            (if (= k (inc words))
              (reduced [result (inc words)])
              [result (inc words)])
            [result words]))]
    (reduce count-words [s 0] (range len))))
(map (partial apply truncate-sentence) ['("Hello how are you Contestant" 4) '("What is the solution to this problem" 4) '("chopper is not a tanuki" 5)])

;;1822
(defn array-sign1 [nums]
  (let [count-signs (fn [[zeros negatives] num]
                      (let [zeros (if (zero? num) (inc zeros) zeros)
                            negatives (if (neg? num) (inc negatives) negatives)]
                        [zeros negatives]))
        [zeros negatives] (reduce count-signs [0 0 0] nums)]
    (cond
      (> zeros 0) 0
      (even? negatives) 1
      :else -1)))
(defn array-sign [nums]
  (let [get-sign (fn [sign num]
            (cond
              (zero? num) (reduced 0)
              (neg? num) (- 0 sign)
              :else sign))]
  (reduce get-sign 1 nums)))
(map array-sign [[-1,-2,-3,-4,3,2,1] [1,5,0,2,-3] [-1,1,-1,1,-1]])

;;1826
(defn bad-sensor [sensor1 sensor2]
  (let [check-sensor (fn [good bad]
                       (let [part2 (vec (drop-last bad))
                             check (fn [result index]
                                     (let [part1 (concat (subvec good 0 index) (subvec good (inc index)))]
                                       (if (= part1 part2)
                                         (reduced true)
                                         result)))]
                         (reduce check false (range (count good)))))]
    (cond
      (and (check-sensor sensor1 sensor2) (not (check-sensor sensor2 sensor1))) 2
      (and (check-sensor sensor2 sensor1) (not (check-sensor sensor1 sensor2))) 1
      :else -1)))
(map (partial apply bad-sensor) ['([2 3 4 5] [2 1 3 4]) '([2 2 2 2 2] [2 2 2 2 5]) '([2 3 2 2 3 2] [2 3 2 3 2 7])])

;;1827
(defn min-operations [nums]
  (let [indexes (range 1 (count nums))
        count-operations (fn [[sum prev] index]
                           (let [num (nums index)]
                             (if (< prev num)
                               [sum num]
                               [(+ sum (inc (- prev num))) (inc prev)])))]
    (first  (reduce count-operations [0 (first nums)] indexes))))
(map min-operations [[1,1,1] [1,5,2,4,1] [8]])

;;1832
(defn check-if-pangram [sentence]
  (->> (vec sentence)
       (set)
       (count)
       (#(= % 26))))
(map check-if-pangram ["thequickbrownfoxjumpsoverthelazydog" "leetcode"])

;;1837
(defn sum-base [n k]
  (letfn [(sum-integer-digits [n k]
            (if (< n k)
              n
              (+ (rem n k) (sum-integer-digits (quot n k) k))))]
    (sum-integer-digits n k)))
(map (partial apply sum-base) ['(34 6) '(10 10)])

;;1844
(defn replace-digits [s]
  (let [cs (vec s)
        shift (fn [c x]
                (char (+ (int c) x)))
        shift-digit (fn [index]
                      (let [c (cs (dec index))
                            x (- (int (cs index)) (int \0))]
                        (shift c x)))
        replace-digit (fn [index]
                        (if (even? index)
                          (cs index)
                          (shift-digit index)))]
    (str/join "" (map replace-digit (range (count cs))))))
(map replace-digits ["a1c1e1" "a1b2c3d4e"])

;;1848
(defn get-min-distance [nums target start]
  (let [abs (fn [n] (if (neg? n) (- 0 n) n))
        get-distance (fn [index]
                       (abs (- index start)))]
    (->> (filter #(= (nums %) target) (range (count nums)))
         (map get-distance)
         (apply min))))
(map (partial apply get-min-distance)
     ['([1,2,3,4,5] 5 3)
      '([1] 1 0)
      '([1,1,1,1,1,1,1,1,1,1] 1 0)])

;;1854
(defn maximum-population [logs]
 (let [year-population-map (->> (map #(apply range %) logs)
       (flatten)
       (frequencies))
       max-population (apply max (vals year-population-map))
       years (sort (keys year-population-map))]
   (reduce (fn [result year]
             (if (= max-population (get year-population-map year))
               (reduced year)
               result)) nil years)))
(map maximum-population [[[1993 1999] [2000 2010]] [[1950 1961] [1960 1971] [1970 1981]]])

;;1859
(defn sort-sentence [s]
  (let [->word-index-pair (fn [word]
                            (let [len (count word)]
                              [(subs word 0 (dec len)) (subs word (dec len) len)]))
        compare-index (fn [l r]
                        (compare (last l) (last r)))]
    (->> (re-seq #"\w+" s)
         (map ->word-index-pair)
         (sort compare-index)
         (map first)
         (str/join " "))))
(map sort-sentence ["is2 sentence4 This1 a3" "Myself2 Me1 I4 and3"])

;;1863
(defn subset-xor-sum [nums]
  (letfn [(xor-sum [index sum]
            (if (= index (count nums))
              sum
              (let [sum1 (xor-sum (inc index) (bit-xor sum (nums index)))
                    sum2 (xor-sum (inc index) sum)]
                (+ sum1 sum2))))]
    (xor-sum 0 0)))
(map subset-xor-sum [[1 3] [5 1 6] [3 4 5 6 7 8]])

;;1869
(defn check-zero-ones [s]
  (let [zeros (re-seq #"0+" s)
        ones (re-seq #"1+" s)
        get-max-length (fn [xs]
                         (if (empty? xs)
                           0
                           (apply max (map count xs))))]
    (> (get-max-length ones) (get-max-length zeros))))
(map check-zero-ones ["1101" "111000" "110100010"])

;;1876
(defn count-good-substring [s]
  (let [good-substring? (fn [index]
                           (let [s' (subs s index (+ index 3))]
                             (= 3 (count (set (vec s'))))))
        indexes (range (- (count s) 2))]
   (count (filter good-substring? indexes))))
(map count-good-substring ["xyzzaz" "aababcabc"])

;;1880
(defn is-sum-equal [first-word second-word target-word]
  (let [->integer (fn [s]
                    (->> (map #(- (int %) (int \a)) (vec s))
                         (str/join "")
                         (Integer/parseInt)))
        sum (+ (->integer first-word) (->integer second-word))
        target (->integer target-word)]
    (= sum target)))
(map (partial apply is-sum-equal) ['("acb" "cba" "cdb") '("aaa" "a" "aab") '("aaa" "a" "aaaa")])

;;1886
(defn find-rotation [mat target]
  (let [rotate (fn [m]
                 (let [row (count m)
                       col row
                       m' (make-array Long/TYPE row col)]
                   (doseq [r (range row) c (range col)]
                     (let [r' (- (dec row) c)
                           c' r]
                       (aset m' r' c' ((m r) c))))
                   (mapv vec m')))
        can-ratate? (fn [[result m] _]
                         (let [m' (rotate m)]
                           (if (= m' target)
                             (reduced [true m'])
                             [result m'])))]
    (first (reduce can-ratate? [false mat] (range 4)))))
(map (partial apply find-rotation) ['([[0 1] [1 0]] [[1 0] [0 1]]) '([[0 1] [1 1]] [[1 0] [0 1]]) '([[0 0 0] [0 1 0] [1 1 1]] [[1 1 1] [0 1 0] [0 0 0]])])

;;1893
(defn is-covered1 [ranges left right]
  (let [to-set (fn [[start end]]
                 (set (range start (inc end))))
        covered? (fn [[start end]]
                   (not-empty (set/intersection (to-set [start end]) (to-set [left right]))))]
    (every? covered? ranges)))
(defn is-covered [ranges left right]
  (let [covered? (fn [num]
                   (reduce (fn [result [start end]]
                             (if (and (>= num start) (<= num end))
                               (reduced true)
                               result)) false ranges)
                   )]
    (every? covered? (range left (inc right)))))
(map (partial apply is-covered) ['([[1 2] [3 4] [5 6]] 2 5) '([[1 10] [10 20]] 21 21)])

;;1897
(defn make-equal [words]
  (let [len (count words)
        freqs (frequencies (flatten (map vec words)))
        counts (vals freqs)]
    (every? #(zero? (rem % len)) counts)))
(map make-equal [["abc" "aabc" "bc"] ["ab" "a"]])

;;1903
(defn largest-odd-number [num]
  (let [digits (vec num)
        len (count digits)
        odd-digit? (fn [c] (odd? (- (int c) (int \0))))]
    (reduce (fn [result index]
              (if (odd-digit? (digits index))
                (reduced (subs num 0 (inc index)))
                result)) "" (reverse (range len)))))
(map largest-odd-number ["52" "4206" "35427"])

;;1909
(defn can-be-increasing [nums]
  (let [increasing? (fn [nums]
                      (apply < nums))
        len (count nums)
        indexes (range 1 (dec len))]
        (letfn [(check-increasing [result index]
                           (let [x0 (nums (dec index))
                                 x1 (nums index)
                                 x2 (nums (inc index))]
                             (cond
                               (and (< x0 x1) (< x1 x2)) result
                               (and (>= x1 x2) (< x0 x2)) (reduced (increasing? (subvec nums (inc index))))
                               :else (reduced false))))]
    (if (or (increasing? (rest nums))
            (increasing? (drop-last nums)))
      true
      (reduce check-increasing true indexes)))))
(map can-be-increasing [[1 2 10 5 7] [2 3 1 2] [1 1 1] [1 2 3] [1 1] [962 23 27 555] [449 354 508 962] [100 21 100] [262 138 583] [1 2 5 10 7]])

;;1913
(defn max-product-difference [nums]
  (let [get-max2-min2 (fn [[max1 max2 min2 min1] num]
            (cond
              (>= num max1) [num max1 min2 min1]
              (> num max2) [max1 num min2 min1]
              (<= num min1) [max1 max2 min1 num]
              (< num min2) [max1 max2 num min1]
              :else [max1 max2 min2 min1])
            )
  [a b c d] (reduce get-max2-min2 (vec (sort > (take 4 nums))) (drop 4 nums))]
    (- (* a b) (* c d))))
(map max-product-difference [[5 6 2 7 4] [4 2 5 9 7 4 8]])

1920
(defn build-arrays [nums]
  (map #(nums %) nums)
)
(defn build-arrays [nums]
  (let [xs (into-array nums)
        low-bits #(bit-and % 0x03ff)
        high-bits #(bit-shift-right % 10)
        ]
    (doseq [index (range (count xs))]
      (let [index' (bit-and (aget xs index) 0x3ff)
            num' (bit-or (bit-shift-left (low-bits (aget xs index')) 10) index')]
        (aset xs index num')))
    (doseq [index (range (count xs))]
      (aset xs index (high-bits (aget xs index))))
    (vec xs)
    ))
(map build-arrays [[0 2 1 5 3 4] [5 0 1 2 3 4]])

;;1925
(defn count-triples [n]
  (let [triples (for [a (range 1 (dec n)) b (range (inc a) n) c (range (inc b) (inc n))]
                  [a b c])
        square (fn [n] (* n n))
        square-sum? (fn [[a b c]]
                     (= (+ (square a) (square b))
                                  (square c)))]
    (* (count (filter square-sum? triples)) 2)))
(map count-triples [5 10])

;;1929
(defn get-concatenation [nums]
  (reduce conj nums nums))
(map get-concatenation [[1 2 1] [1 3 2 1]])

;;1933
(defn is-decomposable [s]
  (let [cs (vec s)]
    (letfn [(decomposable? [cs cnt]
              (cond
                (empty? cs) (= cnt 1)
                (= (count cs) 2) (and (= (first cs) (last cs)) (zero? cnt))
                (apply = (subvec cs 0 3)) (decomposable? (subvec cs 3) cnt)
                (apply = (subvec cs 0 2)) (if (> cnt 0)
                                            (reduced false)
                                            (decomposable? (subvec cs 2) (inc cnt)))
                :else false))]
      (if (= (rem (count s) 3) 2)
        (decomposable? cs 0)
        false))))
(map is-decomposable ["000111000" "00011111222" "011100022233"])

;;1935
(defn can-be-typed-words [text broken-letters]
  (let [broken-letter-set (set (vec broken-letters))
        typable? (fn [word]
                   (empty? (set/intersection
                            (set (vec word))
                            broken-letter-set)))]
    (->> (re-seq #"\w+" text)
         (filter typable?)
         (count))))
(map (partial apply can-be-typed-words) ['("hello world" "ad") '("leet code" "lt") '("leet code" "e")])

;;1941
(defn are-occurrence-equal [s]
  (->> (vec s)
       (frequencies)
       (vals)
       (set)
       (#(= (count %) 1))))
(map are-occurrence-equal ["abacbc" "aaabb"])

;;1945
(defn get-lucky [s k]
  (let [->int #(- (int %) (dec (int \a)))
        transform (fn [s _]
                    (->> (map #(- (int %) (int \0)) (vec s))
                         (apply +)
                         (str)))]
    (->> (map ->int (vec s))
         (str/join "")
         (#(reduce transform % (range k)))
         (Integer/parseInt))))
(map (partial apply get-lucky) ['("iiii" 1) '("leetcode" 2) '("zbax" 2)])

;;1952
(defn is-three [n]
  (let [square (fn [n] (* n n))
        root (int (Math/sqrt n))
        divisible? (fn [a b] (zero? (rem a b)))
        prime? (fn [n]
                 (cond
                   (zero? n) false
                   (= n 1) false
                   (= n 2) true
                   :else (reduce (fn [result i]
                                   (if (divisible? n i)
                                     (reduced false)
                                     result))
                                   true (range 2 (inc (int (Math/sqrt n)))))))]
  (and (= n (square root)) (not= n root) (prime? root))))
(map is-three [2 4])

;;1957
(defn make-fancy-string [s]
  (let [len (count s)
        cs (vec s)
        add-chars (fn [[result start] index]
                    (if (or (= index len) (not= (cs (dec index)) (cs index)))
                      [(str result (subs s start (min index (+ start 2)))) index]
                      [result start]))]
    (first (reduce add-chars ["" 0] (range 1 (inc len))))))
(map make-fancy-string ["leeetcode" "aaabaaaa" "aab"])

;;1961
(defn is-prefix-string [s words]
  (let [prefix? (fn [[result prefix] word]
                  (let [prefix' (str prefix word)]
                    (cond
                      (= s prefix') (reduced [true prefix'])
                      (str/starts-with? s prefix') [result prefix']
                      :else (reduced [false prefix']))))]
    (first (reduce prefix? [false ""] words))))
(map (partial apply is-prefix-string) ['("iloveleetcode" ["i" "love" "leetcode" "apples"]) '("iloveleetcode" ["apples" "i" "love" "leetcode"])])

;;1967
(defn num-of-strings [patterns word]
  (->> (map (partial str/includes? word) patterns)
       (filter true?)
       (count)))
(map (partial apply num-of-strings) ['(["a" "abc" "bc" "d"] "abc") '(["a" "b" "c"] "aaaaabbbbb") '(["a" "a" "a"] "ab")])

;;1971
(defn valid-path [n edges start end]
  (let [add-path (fn [m v1 v2]
                   (let [vertices (or (get m v1) #{})]
                     (assoc m v1 (conj vertices v2))))
        add-paths (fn [m [v1 v2]]
                    (add-path (add-path m v1 v2) v2 v1))
        path-map (reduce add-paths {} edges)]
    (letfn [(find-path [vertices vertice]
              (let [next-vertices (get path-map vertice)
                    find-path' (fn [vertices next-vertices]
                                 (->> (filter #(not (contains? vertices %)) next-vertices)
                                      (map #(find-path (conj vertices %) %))
                                      (reduce #(or %1 %2) false)))]
                (cond
                  (empty? next-vertices) false
                  (contains? next-vertices end) true
                  :else (find-path' vertices next-vertices))))]
      (find-path #{start} start))))
(map (partial apply valid-path) ['(3 [[0 1] [1 2] [2 0]] 0 2) '(6 [[0 1] [0 2] [3 5] [5 4] [4 3]] 0 5)])

;;1974
(defn min-time-to-type [word]
  (let [abs (fn [n]
              (if (neg? n)
                (- 0 n)
                n))
        get-distance (fn [a b]
                       (let [d1 (abs (- (int a) (int b)))
                             d2 (abs (- 26 d1))]
                       (min d1 d2)))
        sum-time (fn [[sum prev] letter]
                   (let [sum' (+ sum 1 (get-distance prev letter))]
                     [sum' letter]))]
    (->> (vec word)
         (reduce sum-time [0 \a])
         (first))))
(map min-time-to-type ["abc" "bza" "zjpc"])

;;1979
(defn find-gcd [nums]
  (let [a (apply max nums)
        b (apply min nums)
        gcd (fn [a b]
              (loop [a1 a b1 b]
                (if (zero? (rem a1 b1))
                  b1
                  (recur b1 (quot a1 b1)))))]
    (gcd a b)))
(map find-gcd [[2 5 6 9 10] [7 5 6 8 3] [3 3]])

;;1984
(defn minimum-difference [nums k]
  (let [nums (vec (sort nums))
        len (count nums)
        find-min-value (fn [min-value index]
                         (let [nums' (subvec nums index (+ index k))]
                         (min min-value (- (last nums') (first nums')))))]
  (->> (range (- (inc len) k))
       (reduce find-min-value (Integer/MAX_VALUE)))))
(map (partial apply minimum-difference) ['([90] 1) '([9 4 1 7] 2)])

;;1991
;; (defn find-middle-index [nums]
;;   (let [sum (apply + nums)
;;         find-index (fn [[result s] index]
;;                      (let [found? (fn [index]
;;                                     (= (* s 2) (- sum (nums index))))]
;;                        (if (found? index)
;;                          (reduced [index s])
;;                          [result (+ s (nums index))])))]
;;     (->> (range (count nums))
;;          (reduce find-index [-1 0])
;;          (first))))
;; (map find-middle-index [[2 3 -1 8 4] [1 -1 4] [2 5] [1]])

;;1995
(defn count-quadruplets1 [nums]
  (let [len (count nums)
        quadruplets (for [a (range (- len 3)) b (range (inc a) (- len 2)) c (range (inc b) (- len 1)) d (range (inc c) len)]
                  [(nums a) (nums b) (nums c) (nums d)])
        quadruplet? (fn [[a b c d]]
                      (= (+ a b c) d))]
    (->> (filter quadruplet? quadruplets)
         (count))))
(defn count-quadruplets [nums]
  (let [len (count nums)
        count-quadruplet (fn [[cnt m] c]
                           (let [d (inc c)
                                 new-map (assoc m (nums d) (inc (or (get m (nums d)) 0)))
                                 indexes (for [b (range (dec c) 0 -1) a (range (dec b) -1 -1)]
                                           [a b])
                                 count-quadruplet' (fn [m cnt [a b]]
                                                     (let [sum (+ (nums a) (nums b) (nums c))]
                                                       (if (nil? (get m sum))
                                                         cnt
                                                         (inc cnt))))
                                 new-count (reduce (partial count-quadruplet' new-map) cnt indexes)]
                             [new-count new-map]))]
    (reduce count-quadruplet [0 {(last nums) 1}] (range (- len 2) 1 -1))))
(map count-quadruplets [[1 2 3 6] [3 3 6 4 5] [1 1 1 3 5]])

;;2000
;; (defn reverse-prefix [word ch]
;;   (let [index (.indexOf word ch)
;;         reverse-prefix' (fn [index]
;;                           (let [part1 (str/reverse (subs word 0 (inc index)))
;;                                 part2 (subs word (inc index))]
;;                             (str part1 part2)))]

;;     (if (= index -1)
;;       word
;;       (reverse-prefix' index))))
;; (map (partial apply reverse-prefix) ['("abcdefd" "d") '("xyxzxe" "z") '("abcd" "z")])

;;2006
(defn count-k-difference1 [nums k]
  (let [abs (fn [n] (if (neg? n) (- n) n))
        len (count nums)
        pairs (for [i (range (dec len)) j (range (inc i) len)]
                [(nums i) (nums j)])
        get-difference (fn [[a b]]
                         (abs (- a b)))]
    (->> (map get-difference pairs)
         (filter #(= % k))
         (count))))
(defn count-k-difference2 [nums k]
  (let [count-pairs (fn [[num-pairs m] index]
                      (let [num (nums index)
                            indexes1 (get m (- num k))
                            indexes2 (get m (+ num k))
                            new-num-pairs (if (or indexes1 indexes2)
                                            (+ num-pairs (+ (count indexes1) (count indexes2)))
                                            num-pairs)
                            new-map (assoc m num (conj (or (get m num) #{}) index))]
                        [new-num-pairs new-map]))
        indexes (range (count nums))]
  (first (reduce count-pairs [0 {}] indexes))))
(defn count-k-difference [nums k]
  (let [counter (make-array Integer/TYPE 101)
        count-pairs (fn [sum num]
                      (let [cnt1 (aget counter num)
                            cnt2 (aget counter (- num k))]
                      (+ sum (* cnt1 cnt2))))]
    (doseq [num nums]
      (aset counter num (inc (aget counter num))))
    (reduce count-pairs 0 (range (inc k) 101))))
(map (partial apply count-k-difference) ['([1 2 2 1] 1) '([1 3] 3) '([3 2 1 5 4] 2)])

;;2011
(defn final-value-after-operations [operations]
  (->> (map #(if (= (subs %1 1 2) "+") 1 -1) operations)
       (apply +)))
(map final-value-after-operations [["--X" "X++" "X++"] ["++X" "++X" "X++"] ["X++" "++X" "--X" "X--"]])

;;2016
(defn maximum-difference1 [nums]
  (let [len (count nums)
        differences (for [i (range (dec len)) j (range (inc i) len)]
                      (- (nums j) (nums i)))]
   (->> (filter pos? differences)
        (#(if (empty? %) -1 (apply max %))))))
(defn maximum-difference [nums]
  (reduce (fn [[max-value start] num]
            (if (< start num)
              [(max max-value (- num start)) start]
              [max-value num])) [0 (first nums)] (rest nums)))
(map maximum-difference [[7 1 5 4] [9 4 3 2] [1 5 2 10]])

;;2022
(defn construct-2d-array [original m n]
  (let [len (count original)
        construct (fn [m n]
                    (let [matrix (make-array Long/TYPE m n)
                          indexes (for [index (range len)]
                                    [index (quot index n) (rem index n)])]
                      (doseq [[index r c] indexes]
                        (aset matrix r c (original index)))
                      (mapv vec matrix)))]
    (if (not= len (* m n))
      []
      (construct m n))))
(map (partial apply construct-2d-array) ['([1 2 3 4] 2 2) '([1 2 3] 1 3) '([1 2] 1 1) '([3] 1 2)])

;;2027
(defn minimum-moves [s]
  (let [digits (mapv #(if (= % \X) 1 0) (vec s))]
    (letfn [(move [digits]
              (let [index (.indexOf digits 1)]
                (cond
                  (= index -1) 0
                  (>= (+ index 3) (count digits)) 1
                  :else (inc (move (subvec digits (+ index 3)))))))]
     (move digits))))
(map minimum-moves ["XXX" "XXOX" "OOOO"])

;;2032
(defn two-out-of-three [nums1 nums2 nums3]
  (let [intersection (fn [nums1 nums2]
                       (set/intersection (set nums1) (set nums2)))
        s1 (intersection nums1 nums2)
        s2 (intersection nums2 nums3)
        s3 (intersection nums3 nums1)]
    (distinct (concat s1 s2 s3))))
(map (partial apply two-out-of-three) ['([1 1 3 2] [2 3] [3]) '([3 1] [2 3] [1 2]) '([1 2 2] [4 3 3] [5])])

;;2037
(defn min-moves-to-seat [seats students]
  (let [seats (sort seats)
        students (sort students)
        abs (fn [n] (if (neg? n) (- n) n))]
(apply +  (map #(abs (- %1 %2)) seats students))))
(map (partial apply min-moves-to-seat) [[[3 1 5] [2 7 4]] [[4 1 5 9] [1 3 2 6]] [[2 2 6 6] [1 3 2 6]]])

;2042
(defn are-numbers-ascending [s]
  (->> (re-seq #"\d+" s)
       (map #(Integer/parseInt %))
       (apply <)))
(map are-numbers-ascending ["1 box has 3 blue 4 red 6 green and 12 yellow marbles" "hello world 5 x 5" "sunset is at 7 51 pm overnight lows will be in the low 50 and 60 s" "4 5 11 26"])

;;2047
(defn count-valid-words [sentence]
  (let [lowercase-hyphen? (fn [token]
                             (every? #(not (Character/isDigit %)) (vec token)))
        at-most-1-hyphen? (fn [token]
                            (let [cs (vec token)
                                  len (count token)
                                  left (.indexOf token "-" )
                                  right (.lastIndexOf token "-")]
                              (or (= left -1)
                                  (and (= left right)
                                       (> left 0) (< left (dec len))
                                       (Character/isLetter (cs (dec left)))
                                       (Character/isLetter (cs (inc left)))))))

        at-most-1-punctuation-mark? (fn [token]
                                      (let [at-most-1-punctuation-mark?' (fn [mark]
                                                                           (let [index (.indexOf token (str mark))]
                                                                             (or (= index -1) (= index (dec (count token))))))]
                                        (every? at-most-1-punctuation-mark?' (vec ",!."))))
        valid-token? (fn [token]
                       (and (lowercase-hyphen? token)
                            (at-most-1-hyphen? token)
                            (at-most-1-punctuation-mark? token)))]
    (->> (str/split sentence #"\s+")
         (filter valid-token?)
        (count))))
(map count-valid-words ["cat and  dog" "!this  1-s b8d!" "alice and  bob are playing stone-game10" "he bought 2 pencils, 3 erasers, and 1  pencil-sharpener."])

;;2053
(defn kth-distinct [arr k]
  (let [find-kth-distinct (fn [[result word-set n] word]
            (if (not (contains? word-set word))
              (if (= n 1)
                (reduced [word word-set 0])
                [result (conj word-set word) (dec n)])
              [result word-set n]))]
  (first (reduce find-kth-distinct ["" #{} k] arr))))
(map (partial apply kth-distinct) ['(["d" "b" "c" "b" "c" "a"] 2) '(["aaa" "aa" "a"] 1) '(["a" "b" "a"] 3)])

;;2057
(defn smallest-equal [nums]
  (let [find-index (fn [result index]
                     (if (= (nums index) (rem index 10))
                       (reduced index)
                       result))]
  (reduce find-index -1 (range (count nums)))))
(map smallest-equal [[0 1 2] [4 3 2 1] [1 2 3 4 5 6 7 8 9 0] [2 1 3 5 2]])
