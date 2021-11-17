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

;;18
(defn four-sum [nums target]
  (let [len (count nums)
        pairs (for [a (range 0 (- len 3)) b (range (inc a) (- len 2))]
                [a b])
        two-sum (fn [target nums]
                  (let [search (fn [[result num-map] index]
                                 (let [num (nums index)
                                       indexes (get num-map num)]
                                   (if (empty? indexes)
                                     [result (assoc num-map (- target num) (conj (get num-map (- target num)) index))]
                                     [(vec (concat result (map (fn [index] [(nums index) num]) indexes))) num-map])))]
                    (first (reduce search [[] {}] (range (count nums))))))
        search-four-sum (fn [result [a b]]
                       (let [x0 (nums a)
                             x1 (nums b)
                             target' (- target x0 x1)
                             nums' (subvec nums (inc b))
                             pairs (two-sum target' nums')]
                         (if (empty? pairs)
                           result
                           (concat result (map #(concat [x0 x1] %) pairs)))))]
    (distinct (reduce search-four-sum [] pairs))))
(map (partial apply four-sum) ['([1 0 -1 0 -2 2] 0) '([2 2 2 2 2] 8)])

;;22
(defn generate-parenthesis1 [n]
  (letfn [(add-parenthesis [s]
            [(str "()" s) (str "(" s ")") (str s "()")])
          (generate [n]
            (if (= n 1)
              ["()"]
              (flatten (map add-parenthesis (generate (dec n))))))]
    (distinct (generate n))))
(defn generate-parenthesis [n]
  (letfn [(generate [result s open close maximum]
            (if (= (count s) (* maximum 2))
              (conj result s)
              (let [r1 (if (< open maximum)
                         (generate result (str "(" s) (inc open) close maximum) result)
                    r2 (if (< close open)
                         (generate r1 (str s ")") open (inc close) maximum) r1)]
                r2)))]
    (generate [] "" 0 0 n)))
(map generate-parenthesis [3 1])

;;29
(defn divide [dividend divisor]
  (let [sign (if (not= (pos? dividend) (pos? divisor)) -1 1)
        abs (fn [n] (if (neg? n) (- n) n))
        a (abs dividend)
        b (abs divisor)]
    (letfn [(divide' [a b]
              (if (>= a b)
                (let [x (loop [x 0]
                          (if (<= a (bit-shift-left b (bit-shift-left 1 x)))
                            x
                            (recur (inc x))))]
                  (+ (bit-shift-left 1 x) (divide' (- a (bit-shift-left b x)) b)))
                0))]
      (if (and (= dividend Integer/MIN_VALUE) (= divisor -1))
        Integer/MAX_VALUE
        (* (divide' a b) sign)))))
(map (partial apply divide) ['(10 3) '(7 -3) '(0 1) '(1 1) '(25 4)])

;;31
(defn next-permunation [nums]
  (let [xs (into-array nums)
        len (count nums)
        indexes (range (- len 2) -1 -1)
        largest-index (reduce (fn [result index]
                                (if (< (aget xs index) (aget xs (inc index)))
                                  (reduced index)
                                  result)) nil indexes)
        swap-array (fn [i j]
                     (let [tmp (aget xs i)]
                       (aset xs i (aget xs j))
                       (aset xs j tmp)))

        reverse-array (fn [start end]
                        (loop [start start end end]
                          (when (< start end)
                            (swap-array start end))))
        next-permunation' (fn [largest-index]
                            (let [find-pivot-index (fn [result index]
                                                     (if (> (aget xs index) (aget xs largest-index))
                                                       (reduced index)
                                                       result))
                                  pivot-index (reduce find-pivot-index nil (range (dec len) largest-index -1))]
                              (swap-array largest-index pivot-index)
                              (reverse-array (inc largest-index) pivot-index)
                              (vec xs)))]
    (if (nil? largest-index)
      (vec (reverse xs))
      (next-permunation' largest-index))))
(map next-permunation [[1 2 3] [3 2 1] [1 1 5] [1] [0 1 2 5 3 3 0]])

;;33
(defn search1 [nums target]
  (letfn [(binary-search-min-index [left right]
            (let [middle (quot (+ left right) 2)]
              (cond
                (>= left right) left
                (> (nums middle) (nums right)) (binary-search-min-index (inc middle) right)
                :else (binary-search-min-index left (dec middle)))))
          (binary-search [nums left right offset]
            (let [len (count nums)
                  middle (quot (+ left right) 2)
                  real-mid (rem (+ middle offset) len)]
              (cond
                (> left right) -1
                (= (nums real-mid) target) real-mid
                (> (nums real-mid) target) (binary-search nums left (dec middle) offset)
                :else (binary-search nums (inc middle) right offset))))]
    (let [min-index (binary-search-min-index 0 (dec (count nums)))]
      (binary-search nums 0 (dec (count nums)) min-index))))
(map (partial apply search1) ['([4 5 6 7 0 1 2] 0) '([4 5 6 7 0 1 2] 3) '([1] 0)])

;;34
(defn search-range [nums target]
  (letfn [(binary-search [left right]
            (let [mid (quot (+ left right) 2)]
              (cond
                (> left right) nil
                (= (nums mid) target) mid
                (> (nums mid) target) (binary-search left (dec mid))
                :else (binary-search (inc mid) right))))
          (get-range [index]
            (let [get-index (fn [indexes]
                              (reduce #(if (not= (nums %2) target) (reduced %1) %2) index indexes))
                  left  (get-index (range (dec index) -1 -1))
                  right (get-index (range (inc index) (count nums)))]
              [left right]))]
    (let [len (count nums)
          index (binary-search 0 (dec len))]
      (if (nil? index)
        [-1 -1]
        (get-range index)))))
(map (partial apply search-range) ['([5 7 7 8 8 10] 8) '([5 7 7 8 8 10] 6) '([] 0)])

;;36
(defn is-valid-sudoku1 [board]
  (let [transpose (fn [m] (apply mapv vector m))
        board' (transpose board)
        indexes (for [r (range 9) c (range 9)] [r c])
        distinct-digits? (fn [cells]
                          (->> (filter #(not= % ".") cells)
                               (apply distinct?)))
        valid-sub-box? (fn [[r c]]
                         (let [indexes (for [r' (range r (+ r 3)) c' (range c (+ c 3))]
                                         [r' c'])
                               concat-cell (fn [result [r c]]
                                             (conj result ((board r) c)))]
                           (distinct-digits? (reduce concat-cell [] indexes))))
        validate-sub-boxes (fn []
                             (let [indexes (for [r (range 0 9 3) c (range 0 9 3)] [r c])]
                               (every? valid-sub-box? indexes)))
        validate' (fn [r c]
                    (every? distinct-digits? [(board r) (board' c)]))
        validate (fn [result [r c]]
                   (cond
                     (= "." ((board r) c)) result
                     (validate' r c) result
                     :else (reduced false)))]
    (reduce validate (validate-sub-boxes) indexes)
    ))
(defn is-valid-sudoku [board]
  (let [indexes (for [r (range 9) c (range 9)]
                  [r c])
        valid? (fn [[result digit-set] [r c]]
                 (let [value ((board r) c)
                       row-value (str "row=" r " num=" value)
                       col-value (str "col=" c " num=" value)
                       subbox-value (str "subbox=" (quot r 3) "," (quot c 3) " num=" value)
                       contains-invalid-value (reduce #(if (contains? digit-set %2)
                                  (reduced false)
                                  %1) true [row-value col-value subbox-value])
                       ]
                   (cond
                     (= value ".") [result digit-set]
                     contains-invalid-value [true (conj digit-set row-value col-value subbox-value)]
                     :else (reduced [false digit-set]))))]
    (first (reduce valid? [true #{}] indexes))))
(map is-valid-sudoku [[["5" "3" "." "." "7" "." "." "." "."]
                       ["6" "." "." "1" "9" "5" "." "." "."]
                       ["." "9" "8" "." "." "." "." "6" "."]
                       ["8" "." "." "." "6" "." "." "." "3"]
                       ["4" "." "." "8" "." "3" "." "." "1"]
                       ["7" "." "." "." "2" "." "." "." "6"]
                       ["." "6" "." "." "." "." "2" "8" "."]
                       ["." "." "." "4" "1" "9" "." "." "5"]
                       ["." "." "." "." "8" "." "." "7" "9"]]
                      [["8" "3" "." "." "7" "." "." "." "."]
                       ["6" "." "." "1" "9" "5" "." "." "."]
                       ["." "9" "8" "." "." "." "." "6" "."]
                       ["8" "." "." "." "6" "." "." "." "3"]
                       ["4" "." "." "8" "." "3" "." "." "1"]
                       ["7" "." "." "." "2" "." "." "." "6"]
                       ["." "6" "." "." "." "." "2" "8" "."]
                       ["." "." "." "4" "1" "9" "." "." "5"]
                       ["." "." "." "." "8" "." "." "7" "9"]]
                      ])

;;(reduce #(str %1 282 %2) "" [56 57 931 56 57 941])
;; (defn formula [n]
;;  (Math/round (- 16662.00002363106 (* 35005.166716504085 n)
;; (* 25457.250036667832 n n -1) (* 8194.291678808782 n n n) (*
;; 1201.750001849751 n n n n -1) (* 65.54166677202554 n n n n n))))
;; (reduce #(str %1 (+ 282 (* (quot (dec %2) 3) 101)) (formula %2)) "" (range 1 7))

;;38
(defn count-and-say [n]
  (letfn [(split-digits [s]
            (let [cs (vec s)
                  len (count s)
                  split (fn [[result start] index]
                          (cond
                            (= index len) [(conj result (subs s start len)) len]
                            (= (cs (dec index)) (cs index)) [result start]
                            :else [(conj result (subs s start index)) index]))]
              (first (reduce split [[] 0] (range 1 (inc len))))))
          (translate [s]
            (let [groups (split-digits s)
                  ->count-char #(str (count %) (subs % 0 1))]
              (str/join "" (map ->count-char groups))))
          (say [n]
            (reduce (fn [result _] (translate result)) "1" (range (dec n))))]
    (say n)))
(map count-and-say [1 4 5])

;;39
(defn combination-sum [candidates target]
  (let [candidates (vec (sort candidates))
        len (count candidates)]
    (letfn [(backtrack [results result target start]
              (let [backtrack' (fn [results index]
                                 (let [candidate (candidates index)]
                                   (backtrack results (conj result candidate) (- target candidate) index)))]
                (cond
                  (neg? target) results
                  (zero? target) (conj results result)
                  :else (reduce backtrack' results (range start len)))))]
      (backtrack [] [] target 0))))
(map (partial apply combination-sum) ['([2 3 6 7] 7) '([2 3 5] 8) '([2] 1) '([1] 1) '([1] 2)])

;;40
(defn combination-sum2 [candidates target]
  (letfn [(backtrack [results result target start]
            (let [len (count candidates)
                  backtrack' (fn [results index]
                               (let [candidate (candidates index)]
                                 (backtrack results (conj result candidate) (- target candidate) (inc index))))]
              (cond
                (neg? target) results
                (zero? target) (conj results result)
                :else (reduce backtrack' results (range start len)))))]
    (->> (backtrack [] [] target 0)
         (mapv sort)
         (distinct)
         (sort-by str))))

(map (partial apply combination-sum2) ['([10 1 2 7 6 1 5] 8) '([2 5 2 1 2] 5)])

;;43
(defn multiply [num1 num2]
  (let [->digits (fn [s]
                   (vec (reverse (map #(- (int %) (int \0)) (vec s)))))
        add (fn [s1 s2]
              (let [digits1 (->digits s1)
                    digits2 (->digits s2)
                    len1 (count digits1)
                    len2 (count digits2)
                    add-digits (fn [[result carry] index]
                                 (let [digit1 (if (< index len1)
                                                (digits1 index)
                                                0)
                                       digit2 (if (< index len2)
                                                (digits2 index)
                                                0)
                                       sum (+ (+ digit1 digit2) carry)]
                                   [(cons (rem sum 10) result) (quot sum 10)]))
                    [digits carry] (reduce add-digits [(list) 0] (range (max len1 len2)))]
                (->> (if (not= carry 0)
                       (cons carry digits)
                       digits)
                     (map str)
                     (str/join ""))))
        times' (fn [s1 digit]
                 (let [digits1 (->digits s1)
                       len1 (count digits1)
                       digit (Integer/parseInt digit)
                       [result carry] (reduce (fn [[result carry] digit1]
                                                (let [r (+ (* digit digit1) carry)]
                                                  [(cons (rem r 10) result) (quot r 10)])) [(list) 0] digits1)]
                   (->> (if (pos? carry)
                          (cons carry result)
                          result)
                        (map str)
                        (str/join ""))))
        times (fn [s1 s2]
                (let [digits (map str (vec s2))
                      multiply-digits (fn [index digit]
                                        (->> (concat (vec (times' s1 digit)) (take index (cycle [\0])))
                                             (map str)
                                             (str/join "")
                                             ))]
                  (->> (map multiply-digits (range (count digits)) (reverse digits))
                       (reduce add "0")
                       )))]
    (times num1 num2)))
(defn multiply [num1 num2]
  (let [num1 (mapv #(- (int %) (int \0)) (vec num1))
        num2 (mapv #(- (int %) (int \0)) (vec num2))
        len1 (count num1)
        len2 (count num2)
        result (make-array Long/TYPE (+ len1 len2))
        indexes (for [i (range (dec len1) -1 -1) j (range (dec len2) -1 -1)]
                  [i j])]
    (doseq [[i j] indexes]
      (let [index (+ i j 1)
            product (* (num1 i) (num2 j))
            sum (+ (aget result index) product)]
        (aset result index (rem sum 10))
        (aset result (dec index) (+ (aget result (dec index)) (quot sum 10)))))
    (let [digits (vec result)
          index (reduce #(if (zero? (digits %2)) %1 (reduced %2)) 0 (range (count digits)))]
      (->> (if (every? zero? digits)
             [0]
             (subvec digits index))
                (map str)
                (str/join "")))))
(map (partial apply multiply) ['("2" "3") '("123" "456") '("0" "0")])

;;45
(defn jump [nums]
  (letfn [(play [min-jumps jumps start]
            (let [len (count nums)
                  indexes (range (inc start) (min len (+ start (nums start) 1)))
                  play' (fn [result index]
                    (play result (inc jumps) index))]
              (cond
                (>= (+ start (nums start)) (dec len)) (min min-jumps (inc jumps))
                (zero? (nums start)) min-jumps
                :else (reduce play' min-jumps indexes))))]
    (play (dec (count nums)) 0 0)))
(defn jump [nums]
  (let [len (count nums)
        indexes (range len)
        jump' (fn [[jumps farthest end] index]
                (let [farthest' (max farthest (+ index (nums index)))
                      jumps' (if (= index end) (inc jumps) jumps)
                      end' (if (= index end) farthest' end)]
                  (if (>= end (dec len))
                    [jumps farthest end]
                    [jumps' farthest' end'])))]
    (first (reduce jump' [0 0 0] indexes))))
(map jump [[2 3 1 1 4] [2 3 0 1 4] [1]])

;;46
(defn permute [nums]
  (letfn [(permutations [nums]
            (cond
              (= (count nums) 1) [nums]
              :else (reduce concat [] (map (fn [index]
                           (let [nums' (vec (concat (subvec nums 0 index)
                                               (subvec nums (inc index))))]
                          (map #(conj % (nums index)) (permutations nums')))) (range (count nums))))))
          ]
    (permutations nums)))
(defn permute [nums]
  (println "++++++++++")
  (let [nums (vec (sort nums))
        len (count nums)]
    (letfn [(backtrack [result start]
              (let [backtrack' (fn [index]
                                 (println result)
                                 (backtrack (conj result (nums index)) (inc index)))
                    indexes (range start (count nums))]
                (cond
                  (= (count result) (count nums)) [result]
                  :else (reduce concat (map backtrack' indexes)))))]
      (backtrack [] 0))))
(map permute [[1 2 3] [0 1] [1]])

;;47
;; (defn permute-unique [nums]
  
;;   )
;; (map permute-unique [[1 1 2] [1 2 3]])

;;48
(defn rotate1 [matrix]
  (let [n (count matrix)
        mat (make-array Long/TYPE n n)
        abs (fn [n] (if (neg? n) (- n) n))
        decode' (fn [n]
                  (cond
                    (neg? n) n
                    (<= n 1000) n
                    :else (- n 2048)))
        encode (fn [x y]
                 (let [get-original (fn [n]
                                      (decode' (rem n 2048)))
                       x' (get-original x)
                       y' (get-original y)]
                   (+ (* x' 2048) y')))
        decode (fn [value]
                 (mapv decode' [(quot value 2048) (rem value 2048)]))
        rotate-element (fn [r c]
                         (let [r' c
                               c' (- (dec n) r)
                               num1 (aget mat r c)
                               num2 (aget mat r' c')]
                           (aset mat r' c' (encode num1 num2))))]
    (doseq [r (range n) c (range n)]
      (aset mat r c ((matrix r) c)))
    (doseq [r (range n) c (range n)]
      (rotate-element r c))
    (doseq [r (range n) c (range n)]
      (aset mat r c (first (decode (aget mat r c)))))
    (mapv vec mat)))

(defn rotate1 [matrix]
  (->> (reverse matrix)
       (apply mapv vector)))
(map rotate1 [[[1 2 3] [4 5 6] [7 8 9]] [[5 1 9 11] [2 4 8 10] [13 3 6 7] [15 14 12 16]]])

;;49
(defn group-anagrams [strs]
  (let [add-anagram (fn [[anagrams-map freq-map] s]
                      (let [freq (or (get freq-map s) (frequencies s))
                            freq-map' (assoc freq-map s freq)
                            anagrams (or (get anagrams-map freq) [])
                            anagrams-map' (assoc anagrams-map freq (conj anagrams s))]
                        [anagrams-map' freq-map']))]
    (->> (reduce add-anagram [{} {}] strs)
         (first)
         (vals))))
(map group-anagrams [["eat" "tea" "tan" "ate" "nat" "bat"] [""] ["a"]])

;;50
(defn pow1 [x n]
  (letfn [(pow1' [x n]
            (cond
              (zero? n) 1
              (even? n) (pow1' (* x x) (quot n 2))
              :else (* x (pow1' (* x x) (quot n 2)))))]
    (if (pos? n)
      (pow1' x n)
      (/ 1 (pow1' x (- n))))))
(map (partial apply pow1) ['(2.0 10) '(2.1 3) '(2.0 -2)])
