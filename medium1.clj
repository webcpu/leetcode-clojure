(ns leetcode.core
  (:gen-class)
  (:require [clojure.string :as str]
            [clojure.set :as set]))

;;3
(defn length-of-longest-substring [s]
  (let [get-max-length (fn [[max-length letter-set] c]
                         (if (contains? letter-set c)
                           [(max max-length (count letter-set)) #{}]
                           [max-length (conj letter-set c)]))
        [max-length letter-set] (reduce get-max-length [0 #{}] (vec s))]
    (max max-length (count letter-set))))
(map length-of-longest-substring ["abcabcbb" "bbbbb" "pwwkew" ""])

;;5
(defn longest-parlindrome [s]
  (let [len (count s)
        indexes (for [i (range len) j (range (inc i) (inc len))]
                  [i j])
        parlindrome? (fn [parlindrome-set s]
                       (loop [s s]
                         (if (not= (subs s 0 1) (subs s (dec (count s)) (count s)))
                           false
                           (if (or (<= (count s) 3) (contains? parlindrome-set (subs s 1 (dec (count s)))))
                             true
                             (recur (subs s 1 (dec (count s))))))))
        max-parlindrome (fn [[result parlindrome-set] [start end]]
                          (let [s' (subs s start end)]
                            (if (parlindrome? parlindrome-set s')
                              (if (> (- end start) (count result))
                                [s' (conj parlindrome-set s')]
                                [result parlindrome-set])
                              [result parlindrome-set])))]
    (first (reduce max-parlindrome ["" #{}] indexes))))
(defn longest-parlindrome [s]
  (letfn [(extend-parlindrome' [cs n i j]
            (if (and (>= i 0) (< j n))
              (if (= (cs i) (cs j))
                (extend-parlindrome' cs n (dec i) (inc j))
                (if (= (inc i) j)
                  [i 1]
                  [(inc i) (- j i 1)]))
              [(inc i) (- j i 1)]))

          (extend-parlindrome [start len cs n i j]
            (let [[start1 len1] (extend-parlindrome' cs n i j)]
              (if (> len1 len)
                [start1 len1]
                [start len])))
          (longest-parlindrome' [s]
            (let [n (count s)
                  cs (vec s)
                  get-max-length (fn [[start len] index]
                                   (let [[start1 len1] (extend-parlindrome start len cs n index index)
                                         [start2 len2] (extend-parlindrome start len cs n index (inc index))]
                                     (if (> len1 len2)
                                       [start1 len1]
                                       [start2 len2])))
                  indexes (range (dec n))
                  [start length]  (reduce get-max-length [0 1] indexes)]
              (subs s start length)))]
    (if (< (count s) 2)
      s
      (longest-parlindrome' s))))

(defn longest-parlindrome [s]
  (let [len (count s)
        cs (vec s)
        dp (make-array Boolean/TYPE len len)
        indexes (for [i (range len) j (range i len)] [i j])
        init-dp (fn []
                  (doseq [[i j] indexes]
                    (when
                     (or (= i j) (= (inc i) j)) (aset dp i j true)))
                  (doseq [i (range (- len 3) -1 -1) j (range (+ i 2) len)]
                    (aset dp i j (and (aget dp (inc i) (dec j)) (= (cs i) (cs j))))))
        get-max-parlindrome (fn [[start end] [i j]]
                              (let [parlindrome (aget dp i j)
                                    len1 (- end start)
                                    len2 (- j i)]
                                (if (and parlindrome (< len1 len2))
                                  [i j]
                                  [start end])))]
    (init-dp)
    (->> (reduce get-max-parlindrome [0 0] indexes)
         (#(subs s (first %) (inc (last %)))))))
(map longest-parlindrome ["babad" "cbbd" "a" "ac"])

;;6
(defn convert [s num-rows]
  (let [cs (vec s)
        results (make-array String num-rows)
        zigzag (fn [[row delta] index]
                 (aset results row (str (aget results row) (cs index)))
                 (if (zero? (rem row (dec num-rows)))
                   [(+ row (* delta -1)) (* delta -1)]
                   [(+ row delta) delta]))
        convert' (fn [s]
                   (reduce zigzag [0 -1] (range (count s)))
                   (str/join "" results))]
    (if (= num-rows 1)
      s
      (convert' s))))
(map (partial apply convert) ['("PAYPALISHIRING" 3) '("PAYPALISHIRING" 4) '("A" 1)])

;;7
(defn reverse1 [x]
  (let [sign (if (neg? x) -1 1)
        abs (fn [n] (if (neg? n) (- n) n))
        compare-string-digits (fn [s1 s2]
                                (let [len1 (count s1)
                                      len2 (count s2)]
                                  (cond
                                    (> len1 len2) true
                                    (= len1 len2) (> (compare s1 s2) 0)
                                    :else false)))
        overflow? (fn [s sign]
                    (if (neg? sign)
                      (compare-string-digits s "2147483648")
                      (compare-string-digits s "2147483647")))
        ->integer (fn [s]
                    (if (overflow? s sign)
                      0
                      (->> (Integer/parseInt s)
                           (#(* sign %)))))]
    (->> (str (abs x))
         (str/reverse)
         (->integer))))
(map reverse1 [123 -123 120 0])

;;8
(defn my-atoi [s]
  (let [get-digits (fn [result c]
                     (cond
                       (= result "") (if (or (= c \+) (= c \-) (Character/isDigit c))
                                       (str result c)
                                       (reduced result))
                       (not (Character/isDigit c)) (reduced result)
                       :else (str result c)))
        digits (reduce get-digits "" (vec (str/trim s)))
        sign (cond
               (zero? (count digits)) 1
               (= (first digits) \-) -11
               :else 1)
        greater (fn [s1 s2]
                  (let [len1 (count s1)
                        len2 (count s2)]
                    (cond
                      (= len1 len2) (pos? (compare s1 s2))
                      (> len1 len2) true
                      :else false)))
        overflow? (fn [s sign]
                    (if (neg? sign)
                      (greater s (str (bit-shift-left 1 31)))
                      (greater s (str (dec (bit-shift-left 1 31))))))
        atoi (fn [digits sign]
               (cond
                 (zero? (count digits)) 0
                 (overflow? digits sign) (if (neg? sign) (* -1 (bit-shift-left 1 31)) (dec (bit-shift-left 1 31)))
                 :else (* sign (reduce #(+ (* %1 10) (- (int %2) (int \0))) 0 (vec digits)))))]
    (atoi (str/replace digits #"[+-]" "") sign)))
(defn my-atoi [s]
  (let [s (str/trim s)
        cs (vec s)
        [result sign] (reduce (fn [[result sign] index]
                                (let [c (cs index)
                                      digit (- (int c) (int \0))]
                               (cond
                                 (and (zero? index) (= c \+)) [result 1]
                                 (and (zero? index) (= c \-)) [result -1]
                                 (not (Character/isDigit c)) (reduced [result sign])
                                 (or (> result (quot Integer/MAX_VALUE 10))
                                     (and (= result (quot Integer/MAX_VALUE 10))
                                          (> digit 7))) (if (= sign 1)
                                                          (reduced [Integer/MAX_VALUE 1])
                                                          (reduced [Integer/MIN_VALUE 1]))
                                 :else [(+ (* result 10) digit) sign])))
                              [0 1] (range (count s)))]
    (* result sign)
  )
  )
(map my-atoi ["42" "   -42" "4193 with words" "words and 987" "-91283472332"])

;;11
(defn max-area [height]
  (letfn [(max-area' [area1 start end]
            (let [num1 (height start)
                  num2 (height end)
                  area2 (* (- end start) (min num1 num2))
                  area (max area1 area2)]
              (if (>= start end)
                area1
                (if (> num1 num2)
                  (max-area' area start (dec end))
                  (max-area' area (inc start) end)))))]
    (max-area' 0 0 (dec (count height)))))
(map max-area [[1 8 6 2 5 4 8 3 7] [1 1] [4 3 2 1 4] [1 2 1]])

;;12
(defn int-to-roman [num]
  (let [roman-map {1 "I"  2 "II" 3 "III" 4 "IV" 5 "I" 6 "VI" 7 "VII" 8 "VIII" 9 "IX"
                   10 "X" 20 "XX" 30 "XXX" 40 "XL" 50 "L" 60 "LX" 70 "LXX" 80 "LXXX" 90 "XC"
                   100 "C" 200 "CC" 300 "CCC" 400 "CD" 500 "D" 600 "DC" 700 "DCC" 800 "DCCC" 900 "CM"
                   1000 "M" 2000 "MM" 3000 "MMM"}
        digits (->> (vec (str num))
                    (mapv #(- (int %) (int \0))))
        indexes (range (count digits))
        len (count digits)
        to-roman (fn [index] (->> (* (digits index) (int (Math/pow 10 (- len index 1))))
                                  (get roman-map)))]
    (->> (mapv to-roman indexes)
         (str/join ""))))
(defn int-to-roman [num]
  (let [M ["" "M" "MM" "MMM"]
        C ["" "C" "CC" "CCC" "CD" "D" "DC" "DCC" "DCCC" "CM"]
        X ["" "X" "XX" "XXX" "XL" "L" "LX" "LXX" "LXXX" "XC"]
        I ["" "I" "II" "III" "IV" "V" "VI" "VII" "VIII" "IX"]
        romans [M C X I]
        digits [(quot num 1000) (quot (rem num 1000) 100) (quot (rem num 100) 10) (rem num 10)]]
    (->> (map #(get %1 %2) romans digits)
         (str/join ""))))
(map int-to-roman [3 4 9 58 1994])

;;15
(defn three-sum [nums]
  (let [len (count nums)
        two-sum (fn [xs target]
                (first (vec (reduce (fn [[result m] index]
                            (let [num (xs index)
                                  indexes (get m num)
                                  pairs (mapv (fn [i] [num (get xs i)]) indexes)
                                  ]
                              (if (nil? indexes)
                                [result (assoc m (- target num) (conj indexes index))]
                                [(vec (concat result pairs)) m])))
                                    [[] {}] (range (count xs))))))
        add-triplet (fn [result index]
                      (let [num (nums index)
                            target (- num)
                            xs (vec (concat (subvec nums 0 index) (subvec nums (inc index))))
                            twiplets (two-sum xs target)
                            ]
                        (if (empty? twiplets)
                          []
                          (concat result (mapv #(vec (concat [num] %)) twiplets)))))
        ]
    (->> (reduce add-triplet [] (range (- len 2)))
         (map sort)
         (distinct)
         (vec))))
(map three-sum [[-1,0,1,2,-1,-4] [] [0]])

;;16
(defn three-sum-closest [nums target]
  (let [add-pair (fn [target [result m] index]
                   (let [num (nums index)
                         indexes (get m num)]
                     (if (empty? indexes)
                       [result (assoc m (- target num) (conj (get m (- target num)) num))]
                       [(concat result (map #(partial vector num (nums %)) indexes))])))
        two-sum (fn [nums target]
                  (reduce #(add-pair target %1 %2) [[] {}] (range (count nums))))
        nums (vec (sort nums))
        min-target (apply + (take 3 nums))
        max-target (apply + (take 3 (reverse nums)))
        three-sum' (fn [result target]
                     (let [count-three-sum
                           (fn [r num]
                             (let [pairs (two-sum nums target)]
                               (if (empty? pairs)
                                 r
                                 (reduced (count pairs))))
                             )]
                       (reduce count-three-sum 0 nums)))
        three-sum (fn [min-target max-target]
                    (reduce three-sum' 0 (range target (inc max-target)))
                    )]
    (three-sum min-target max-target)))
(map (partial apply three-sum-closest) ['([-1 2 1 -4] 1) '([0 0 0] 1)])

;;17
(defn letter-combinations [digits]
  (let [digit-map {\2 "abc" \3 "def" \4 "ghi" \5 "jkl" \6 "mno" \7 "pqrs" \8 "tuv" \9 "wxyz"}
        digits-list (mapv #(vec (get digit-map %)) digits)]
    (letfn [(permunations [digits-list]
              (let [indexes (range (count digits-list))
                    permunations' (fn [index] (for [digits (permunations (subvec digits-list (inc index)))
                                                    digit (digits-list index)]
                                                (cons digit digits)))]
                (if (= (count digits-list) 1)
                  (map (fn [digit] [digit]) (first digits-list))
                  (reduce concat [] (map permunations' indexes)))))]
      (->> (permunations digits-list)
           (map #(str/join "" %))
           (distinct))
      )))
(map letter-combinations ["23" "2"])
