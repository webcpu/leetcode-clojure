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

(->> (range 0 10)
     (reduce (fn [[evens odds] x]
               (if (even? x)
                 [(conj evens x) odds]
                 [evens (conj odds x)]))
             [[] []]))

;;54
(defn spiral-order [matrix]
  (letfn [(transpose [mat]
            (let [m (count mat)
                  n (count (mat 0))
                  mt (make-array Long/TYPE n m)]
              (doseq [r (range m) c (range n)]
                (aset mt c r ((mat r) c)))
              (mapv vec mt)))
          (rotate-matrix [m]
            (vec (reverse (transpose m))))
          (spiral [mat]
            (if (empty? (subvec mat 1))
              (first mat)
              (vec (concat (first mat) (spiral (rotate-matrix (subvec mat 1)))))))]
    (spiral matrix)))
(map spiral-order [[[1 2 3] [4 5 6] [7 8 9]] [[1 2 3 4] [5 6 7 8] [9 10 11 12]]])

;;55
(defn can-jump [nums]
  (let [xs (vec (reverse nums))
        jump (fn [[result start] index]
               (let [num (xs index)]
                 (cond
                   (zero? num) [result index]
                   (nil? start) [result nil]
                   (> num (- index start)) [result start]
                   :else (reduced [false start]))))
        [result index] (reduce jump [true nil] (range (count nums)))]
    (and result (not= index (dec (count nums))))))
(defn can-jump [nums]
  (let [jump (fn [max-value index]
               (if (<= index max-value)
                 (max max-value (+ index (nums index)))
                 (reduced max-value)))
        max-jump (reduce jump 0 (range (count nums)))]
    (>= max-jump (dec (count nums)))))
(map can-jump [[2 3 1 1 4] [3 2 1 0 4] [0 1 2 3]])

;;56
(defn merge1 [intervals]
  (let [points (make-array Long/TYPE 10002)
        [min-value max-value] (reduce  (fn [[min-value max-value] [start end]]
                                         (doseq [index (range start (inc end))]
                                           (aset points index 1))
                                         [(min min-value start) (max max-value end)]) [Long/MAX_VALUE 0] intervals)
        add-interval (fn [[result start] index]
                       (let [num (aget points index)]
                         (cond
                           (and (nil? start) (= num 1)) [result index]
                           (and (nil? start) (= num 0)) [result start]
                           (= num 0) [(conj result [start (dec index)]) nil]
                           :else [result start])))
        [result start] (reduce add-interval [[] nil] (range min-value (inc max-value)))]
    (if (nil? start)
      result
      (conj result [start max-value]))))
(defn merge1 [intervals]
  (let [intervals (sort-by first intervals)
        add-interval (fn [result [start end]]
          (if (and (not-empty result) (>= (last (last result)) start))
            (conj (vec (drop-last result)) [(first (last result)) end])
            (conj result [start end])))]
    (reduce add-interval [] intervals)))
(map merge1 [[[1 3] [2 6] [8 10] [15 18]] [[1 4] [4 5]]])

;;57
(defn insert1 [intervals new-interval]
  (let [len (count intervals)
        [start end] new-interval
        left (filterv #(< (last %) start) intervals)
        right (filterv #(> (first %) end) intervals)
        intervals' (vec (concat left right))
        interval (if (= intervals intervals')
                   [start end]
                   [(min start (first (intervals (count left))))
                    (max end (last (intervals (- (dec len) (count right)))))])]
    (concat (conj left interval) right)))
(map (partial apply insert1) ['([[1 3] [6 9]] [2 5]) '([[1 2] [3 5] [6 7] [8 10] [12 16]] [4 8]) '([] [5 7]) '([[1 5]] [2 3])])

;;59
(defn generate-matrix [n]
  (letfn [(transpose [matrix]
            (let [m (count matrix)
                  n (count (first matrix))                  ]
              (if (zero? n)
                matrix
                (let [mt (make-array Long/TYPE n m)]
                  (doseq [r (range m) c (range n)]
                    (aset mt c r ((matrix r) c)))
                  (mapv vec mt)))))
          (generate' [matrix size len]
            (vec (concat [(vec (range (- size len) size))]
                         (map #(vec (reverse %)) (transpose matrix)))))
          (generate [matrix size]
            (let [len (count matrix)
                  matrix' (generate matrix size len)]
              (if (> size 1)
                (generate matrix' (- size len))
                matrix)))]
    (generate [[(* n n)]] (* n n))))
(map generate-matrix [3 1])

;;62
(defn unique-paths [m n]
  (let [dp (make-array Long/TYPE m n)]
      (doseq [c (range 1 n)]
        (aset dp 0 c 1))
      (doseq [r (range 1 m)]
        (aset dp r 0 1))
      (doseq [r (range 1 m) c (range 1 n)]
        (let [paths1 (aget dp (dec r) c)
              paths2 (aget dp r (dec c))]
        (aset dp r c (+ paths1 paths2))))
      (aget dp (dec m) (dec n))))
(map (partial apply unique-paths) ['(3 7) '(3 3)])

;;63
(defn unique-paths-with-obstacles [obstacle-grid]
  (let [m (count obstacle-grid)
        n (count (first obstacle-grid))
        dp (make-array Long/TYPE m n)
        init-dp (fn [indexes]
                  (doseq [[r c] indexes]
                    (if (zero? ((obstacle-grid r) c))
                      (aset dp r c 1)
                      (aset dp r c 0))))

        first-row (for [c (range 1 n)]
                    [0 c])
        first-col (for [r (range 1 m)]
                    [r 0])
        calculate-dp (fn []
                       (doseq [r (range 1 m) c (range 1 n)]
                         (let [obstacle ((obstacle-grid r) c)
                               paths1 (aget dp (dec r) c)
                               paths2 (aget dp r (dec c))]
                           (if (= obstacle 0)
                             (aset dp r c (+ paths1 paths2))
                             (aset dp r c 0)))))]
    (init-dp first-row)
    (init-dp first-col)
    (calculate-dp)
    (aget dp (dec m) (dec n))))
(map unique-paths-with-obstacles [[[0 0 0] [0 1 0] [0 0 0]] [[0 1] [0 0]]])

;;64
(defn min-path-sum [grid]
  (let [m (count grid)
        n (count (first grid))
        dp (make-array Long/TYPE m n)
        get-path-sum (fn [sum [r c]]
                       (let [sum' (+ sum ((grid r) c))]
                         (aset dp r c sum')
                         sum'))
        init-dp (fn []
                  (reduce get-path-sum 0 (map (fn [index] [index 0]) (range m)))
                  (reduce get-path-sum 0 (map (fn [index] [0 index]) (range n)))
                  (doseq [r (range 1 m) c (range 1 n)]
                    (let [sum1 (aget dp (dec r) c)
                          sum2 (aget dp r (dec c))
                          num ((grid r) c)
                          path-sum (+ (min sum1 sum2) num)]
                      (aset dp r c path-sum))))]
    (init-dp)
    (aget dp (dec m) (dec n))))
(map min-path-sum [[[1 3 1] [1 5 1] [4 2 1]] [[1 2 3] [4 5 6]]])

;;71
(defn simplify-path [s]
  (let [components (str/split s #"/+")
        simplify (fn [result component]
                   (case component
                     ("" ".") result
                     ".." (vec (drop-last result))
                     (conj result component)))]
    (->> (reduce simplify [] components)
         (str/join "/")
         (#(str "/" %)))))
(map simplify-path ["/home/" "/../" "/home//foo/" "/a/./b/../../c/"])

;;73
(defn set-zeroes [matrix]
  (let [m (count matrix)
        n (count (first matrix))
        mat (make-array Long/TYPE m n)
        rows (filterv (fn [r]
                        (reduce (fn [result c] (if (zero? ((matrix r) c))
                                   (reduced true)
                                   result)) false (range n))) (range m))

        cols (filterv (fn [c]
                        (reduce (fn [result r] (if (zero? ((matrix r) c))
                                                 (reduced true)
                                                 result)) false (range m))) (range n))]
    (doseq [r (range m) c (range n)]
      (aset mat r c ((matrix r) c)))
    (doseq [r rows c (range n)]
      (aset mat r c 0))
    (doseq [r (range m) c cols]
      (aset mat r c 0))
    (mapv vec mat)))
(map set-zeroes [[[1 1 1] [1 0 1] [1 1 1]] [[0 1 2 0] [3 4 5 2] [1 3 1 5]]])

;;74
(defn search-matrix [matrix target]
  (let [m (count matrix)
        n (count (first matrix))]
    (letfn [(binary-search [nums left right]
              (let [mid (quot (+ left right) 2)]
                (cond
                  (> left right) false
                  (> (nums mid) target) (binary-search nums left (dec mid))
                  (< (nums mid) target) (binary-search nums (inc mid) right)
                  :else true)))
            (search [r c]
              (cond
                (or (< r 0) (>= r m) (< c 0) (>= c n)) false
                (= ((matrix r) c) target) true
                (> ((matrix r) c) target) (binary-search (matrix r) 0 c)
                :else (search (inc r) c)
                ))]
      (search 0 (dec n)))))
(defn search-matrix [matrix target]
  (let [m (count matrix)
        n (count (first matrix))]
    (letfn [(search [left right]
              (let [mid (quot (+ left right) 2)
                    num ((matrix (quot mid m)) (rem mid n))]
                (cond
                  (> left right) false
                  (< target num) (search left (dec mid))
                  (> target num) (search (inc mid) right)
                  :else true)))]
    (search 0 (dec (* m n))))))
(map (partial apply search-matrix) ['([[1,3,5,7],[10,11,16,20],[23,30,34,60]] 3) '([[1,3,5,7],[10,11,16,20],[23,30,34,60]] 13)])

;;75
(defn sort-colors [nums]
  (let [result (make-array Long/TYPE (count nums))
        counter (make-array Long/TYPE 3)]
    (doseq [num nums]
      (aset counter num (inc (aget counter num))))

    (doseq [index (range (count nums))]
      (let [color (cond
                    (< index (aget counter 0)) 0
                    (and (<= (aget counter 0) index) (< index (+ (aget counter 0) (aget counter 1)))) 1
                    :else 2)]
        (aset result index color)))
    (vec result)))
(map sort-colors [[2 0 2 1 1 0] [2 0 1] [0] [1]])

;;77
(defn combine [n k]
  (letfn [(combine' [nums k]
            (if (= k 1)
              (set (map #(set [%]) nums))
              (reduce (fn [result num]
                        (set/union result (set (map #(conj % num) (combine' (disj nums num) (dec k))))))
                      #{} nums)
              ))]
    (mapv vec (combine' (set (range 1 (inc n))) k))))
(map (partial apply combine) ['(4 2) '(1 1)])

;;78
(defn subsets [nums]
  (letfn [(subset [nums k]
            (let [subset' (fn [num]
                            (set (map #(conj % num) (subset (disj nums num) (dec k)))))]
            (cond
              (zero? k) #{}
              (= k 1) (set (map #(set (vector %)) nums))
              :else (set (apply concat (map subset' nums))))))]
    (->> (map #(subset (set nums) %) (range (inc (count nums))))
         (apply concat)
         (mapv vec)
         (#(conj % [])))))
(map subsets [[1 2 3] [0]])

;;79
(defn exist [board word]
  (let [m (count board)
        n (count (first board))
        indexes (for [r (range m) c (range n)]
                  [r c])
        board-map (reduce (fn [result [r c]]
                            (let [letter ((board r) c)]
                              (assoc result letter (conj (get result letter) [r c]))))
                          {} indexes)
        valid? (fn [path [r c]]
                 (and (>= r 0) (>= c 0) (< r m) (< c n) (not (contains? path [r c]))))]
    (letfn [(search [path letters]
              (let [letter (first letters)
                    indexes (filter #(valid? path %) (get board-map letter))
                    neighbours (fn [[r c]]
                                 (filter #(valid? path %) [[(inc r) c] [(dec r) c] [r (inc c)] [r (dec c)]]))
                    search' (fn [result [r c]]
                              (let [result (reduce (fn [found x]
                                                     (let [r (search (conj path [r c]) (rest letters))]
                                                       (if r
                                                         (reduced true)
                                                         found)
                                                       ))
                                                   false
                                                   (neighbours [r c]))]
                                (if result
                                  (reduced true)
                                  result)))
                    ]
                (cond
                  (empty? letters) true
                  (empty? indexes) false
                  :else (reduce search' false indexes))))]
      (search #{} (map str (vec word)))
      )))

(defn exist [board word]
  (let [len (count word)
        m (count board)
        n (count (first board))
        indexes (for [r (range m) c (range n)]
                  [r c])
        letters (map str (vec word))
        ]
    (letfn [(search [board-status letters r c i]
              (cond
                (= i len) true
                (or (neg? r) (neg? c) (>= r m) (>= c n) (aget board-status r c)) false
                (not= ((board r) c) (first letters)) false
                :else (do
                        (aset board-status r c true)
                        (or
                         (search board-status (rest letters) (inc r) c (inc i))
                         (search board-status (rest letters) (dec r) c (inc i))
                         (search board-status (rest letters) r (inc c) (inc i))
                         (search board-status (rest letters) r (dec c) (inc i))))))]
      (reduce (fn [result [r c]]
                (let [board-status (make-array Boolean/TYPE m n)]
                  (if (search board-status letters r c 0)
                    (reduced true)
                    result))) false indexes))))
(map (partial apply exist) ['([["A" "B" "C" "E"] ["S" "F" "C" "S"] ["A" "D" "E" "E"]] "ABCCED") '([["A" "B" "C" "E"] ["S" "F" "C" "S"] ["A" "D" "E" "E"]] "SEE") '([["A" "B" "C" "E"] ["S" "F" "C" "S"] ["A" "D" "E" "E"]] "ABCB")])

;;80
(defn remove-duplicates [nums]
  (let [xs (into-array nums)]
    (reduce (fn [index num]
              (if (or (< index 2) (> num (aget xs (- index 2))))
                (do
                  (aset xs index num)
                  (inc index))
                index)) 0 nums)))
(map remove-duplicates [[1 1 1 2 2 3] [0 0 1 1 1 1 2 3 3]])

;;81
(defn search [nums target]
  (letfn [(binary-search [left right]
            (let [mid (quot (+ left right) 2)]
              (cond
                (> left right) false
                (= (nums mid) target) true
                (< (nums left) (nums mid)) (if (<= (nums left) target)
                                             (binary-search left (dec right))
                                             (binary-search (inc mid) right))
                (> (nums mid) (nums left)) (if (and (< (nums mid) target)
                                                     (< target (nums right)))
                                              (binary-search (inc mid) right)
                                              (binary-search left (dec mid)))
                :else (binary-search left (dec right))
                )))]
    (binary-search 0 (dec (count nums)))))
(map (partial apply search) ['([2 5 6 0 0 1 2] 0) '([2 5 6 0 0 1 2] 3)])

;;89
(defn gray-code [n]
  (letfn [(generate-gray-code [n]
            (let [generate (fn [result _]
                             (let [len (count result)]
                               (concat result (reverse (map #(+ % len) result)))
                               ))]
            (reduce generate [0 1] (range 1 n))))]
    (generate-gray-code n)))
(map gray-code [2 1 3])

;;90
(defn subsets-with-dup [nums]
  (let [nums (vec (sort nums))]
  (letfn [(subsets [results result nums start]
            (let [results' (conj results result)]
              (reduce (fn [r index]
                        (if (or (= start index) (not= (nums (dec index)) (nums index)))
                          (subsets r (conj result (nums index)) nums (inc index))
                          r))
                         results' (range start (count nums)))
            ))]
    (subsets [] [] nums 0))))
(map subsets-with-dup [[1 2 2] [0]])

;;91
(defn num-decoding [s]
  (let [digits (mapv #(- (int %) (int \0)) (vec s))]
    (letfn [(valid? [digits]
              (cond
                (> (count digits) 2) false
                (zero? (first digits)) false
                :else (< 0 (reduce #(+ (* %1 10) %2) (first digits) (rest digits)) 27)))
            (count-decoding [digits]
              (let [len (count digits)
                    digit1 (first digits)
                    digit2 (second digits)]
                (cond
                  (= len 0) 1
                  (= len 1) (if (valid? digits) 1 0)
                  (= len 2) (+ (if (valid? (take 2 digits)) 1 0) (if (valid? (take 1 digits)) (count-decoding (rest digits)) 0))
                  :else (+ (if (valid? (take 1 digits)) (count-decoding (drop 1 digits)) 0)
                           (if (valid? (take 2 digits)) (count-decoding (drop 2 digits)) 0))
                  )))]
      (count-decoding digits))))

(defn num-decoding [s]
  (let [len (count s)
        dp (make-array Long/TYPE (inc len))
        digits (mapv #(- (int %) (int \0)) (vec s))]
    (aset dp 0 1)
    (aset dp 1 (if (zero? (first digits)) 0 1))
    (doseq [index (range 2 (inc len))]
      (let [digit1 (digits (- index 1))
            digit2 (digits (- index 2))]
        (when (< 0 digit1 10)
          (aset dp index (+ (aget dp index) (aget dp (dec index)))))
        (when (< 9 (+ (* digit2 10) digit1) 27)
          (aset dp index (+ (aget dp index) (aget dp (- index 2)))))))
      (last dp)))
  (map num-decoding ["12" "226" "0" "06" "00"])

;;93
(defn restore-ip-addresses [s]
  (let [min-length (max 1 (rem (- (count s) 9) 4))
        valid? (fn [s]
                  (let [len (count s)
                        digits (map #(- (int %) (int \0)) s)
                        num (reduce #(+ (* %1 10) %2) 0 digits)]
                  (and (not (and (> len 1) (zero? (first digits)))) (<= 0 num 255))))]
    (letfn [(restore [results result start]
              (let [restore' (fn [r end]
                               (let [s' (subs s start end)
                                     num (Integer/parseInt s')]
                                 (if (valid? s')
                                   (restore r (conj result (str num)) end)
                                   r)))]
                (cond
                  (and (= (count result) 4) (= start (count s))) (conj results (str/join "." result))
                  (= (count result) 4) results
                  :else (reduce restore' results (range (inc start) (min (inc (count s)) (+ start 4)))))))]
      (vec (set (restore [] [] 0))))))
(defn restore-ip-addresses [s]
  (let [len (count s)
        to-components (fn [[a b c d]]
                        (let [num1 (subs s 0 a)
                              num2 (subs s a (+ a b))
                              num3 (subs s (+ a b) (+ a b c))
                              num4 (subs s (+ a b c) (+ a b c d))]
                          [num1 num2 num3 num4]
                          ))
        valid? (fn [[num1 num2 num3 num4]]
                 (->> (filter #(not (empty? %)) [num1 num2 num3 num4])
                      (map #(Integer/parseInt %) )
                      (filter #(<= 0 % 255))
                      (map str)
                      (str/join "")
                      (#(= (count %) (count s)))))
        indexes (filter #(= (apply + %) len) (for [a (range 1 4) b (range 1 4) c (range 1 4) d (range 1 4)]
                                               [a b c d]))]
    (->> (map to-components indexes)
         (filter valid?)
         (map #(str/join "." %))
         )))
(map restore-ip-addresses ["25525511135" "0000" "1111" "010010" "101023"])

;;97
(defn is-interleave [s1 s2 s3]
  (let [cs1 (vec s1)
        cs2 (vec s2)
        cs3 (vec s3)]
    (letfn [(interleave? [cs1 cs2 cs3]
              (cond
                (not= (+ (count cs1) (count cs2)) (count cs3)) false
                (every? empty? [cs1 cs2 cs3]) true
                (and (not= (first cs1) (first cs3)) (not= (first cs2) (first cs3))) false
                :else (or (and (= (first cs1) (first cs3)) (interleave? (rest cs1) cs2 (rest cs3)))
                         (and (= (first cs2) (first cs3)) (interleave? cs1 (rest cs2) (rest cs3))))))]
      (interleave? cs1 cs2 cs3))))
(defn is-interleave [s1 s2 s3]
  (let [len1 (count s1)
        len2 (count s2)
        dp (make-array Boolean/TYPE (inc len1) (inc len2))
        interleave? (fn [dp]
                      (doseq [i (range (inc len1)) j (range (inc len2))]
                        (cond
                          (= i j 0) (aset dp i j true)
                          (zero? i) (aset dp i j (and (aget dp i (dec j))
                                                      (= (subs s2 (dec j) j) (subs s3 (dec j) j))))
                          (zero? j) (aset dp i j (and (aget dp (dec i) j)
                                                      (= (subs s1 (dec i) i) (subs s3 (dec i) i))))
                          :else (let [value (or (and (aget dp i (dec j))
                                                     (= (subs s2 (dec j) j) (subs s3 (+ i j -1) (+ i j))))
                                                (and (aget dp (dec i) j)
                                                     (= (subs s1 (dec i) i) (subs s3 (+ i j -1) (+ i j)))))]
                                  (aset dp i j value))))
                      (aget dp len1 len2))]
    (if (= (+ len1 len2) (count s3))
      (interleave? dp)
      false)))
(map (partial apply is-interleave) ['("aabcc" "dbbca" "aadbbcbcac") '("aabcc" "dbbca" "aadbbbaccc") '("" "" "")])

;;120
(defn minimum-total [triangle]
  (let [m (count triangle)
        n (count (first triangle))]
    (letfn [(dfs [sum r c]
              (if (= (inc r) m)
                sum
                (min (dfs (+ sum ((triangle (inc r)) c)) (inc r) c)
                     (dfs (+ sum ((triangle (inc r)) c)) (inc c) (inc c)))))]
      (dfs (first (first triangle)) 0 0))))
(defn minimum-total [triangle]
  (let [m (count triangle)
        dp (into-array (last triangle))
        indexes (for [r (range (- m 2) -1 -1) c (range (inc r))]
                  [r c])]
    (doseq [[r c] indexes]
      (aset dp c (+ (aget dp c)
                    (min (aget dp c) (aget dp (inc c))))))
    (first dp)))
(map minimum-total [[[2] [3 4] [6 5 7] [4 1 8 3]] [[-10]]])

;;122
(defn max-profit [prices]
  (let [make-profit (fn [profit index]
                      (let [price1 (prices (dec index))
                            price2 (prices index)]
                        (if (< price1 price2)
                          (+ (- price2 price1) profit)
                          profit)
                        ))]
    (reduce make-profit 0 (range 1 (count prices)))))
(map max-profit [[7 1 5 3 6 4] [1 2 3 4 5] [7 6 4 3 1]])

;;128
(defn longest-consective [nums]
  (let [num-set (set nums)]
    (letfn [(count-length [num]
              (loop [num num len 0]
                (if (not (contains? num-set num))
                  len
                  (recur (inc num) (inc len)))))
            (longest [max-length num]
              (if (contains? num-set (dec num))
                max-length
                (max (count-length num) max-length)))]
      (reduce longest 1 num-set))))
(map longest-consective [[100,4,200,1,3,2] [0,3,7,2,5,8,4,6,0,1]])

;; (rem (reduce #(let [r (* %1 %2)]
;;            (cond
;;              (zero? (rem r 100)) (quot r 100)
;;              (zero? (rem r 10)) (quot r 10)
;;              :else (rem r 100)
;;            )) 1 (range 1 91)) 100)

;;130
(defn solve [board]
  (let [m (count board)
        n (count (first board))
        indexes (->> (for [r (range m) c (range n)]
                       [r c])
                     (filter (fn [[r c]]
                               (and (or (zero? r) (zero? c) (= r (dec m)) (= c (dec n)))
                                    (= ((board r) c) "O")))))]
    (letfn [(free-cell? [cell-set [r c]]
              (and (>= r 0) (< r m) (>= c 0) (< c n)
                   (= ((board r) c) "O")
                   (not (contains? cell-set [r c]))))
            (get-free-cells [free-cells [r c]]
              (if (not (free-cell? free-cells [r c]))
                free-cells
                (reduce get-free-cells (conj free-cells [r c]) [[(inc r) c] [(dec r) c] [r (inc c)] [r (dec c)]])))]
      (let [free-cells (reduce get-free-cells #{} indexes)
            cells (make-array String m n)]
        (doseq [r (range m) c (range n)]
          (aset cells r c (if (contains? free-cells [r c]) "O" "X")))
        (mapv vec cells)
        ))))
(map solve [[["X" "X" "X" "X"] ["X" "O" "O" "X"] ["X" "X" "O" "X"] ["X" "O" "X" "X"]] [["X"]]])

;;131
(defn partition1 [s]
  (let [len (count s)]
    (letfn [(parlindrome? [s]
              (= s (str/reverse s)))
            (part [results result start]
              (let [add-parlindrome (fn [results end]
                                      (let [s' (subs s start end)]
                                        (if (parlindrome? s')
                                          (part results (conj result s') end)
                                          results)))]
                (if (= start len)
                  (conj results result)
                  (reduce add-parlindrome results (range (inc start) (inc len))))))]
      (part [] [] 0))))
(map partition1 ["aab" "a" "aabaa"])

;;134
(defn can-complete-circuit [gas cost]
  (let [len (count gas)
        total (- (apply + gas) (apply + cost))
        search (fn [[tank start] index]
                 (let [delta (- (gas index) (cost index))
                       tank (+ tank delta)]
                   (if (neg? tank)
                     [0 (inc start)]
                     [tank start])))
        search-gas-station (fn []
                             (last (reduce search [0 0] (range len))))]
    (if (neg? total)
      -1
      (search-gas-station))))
(map (partial apply can-complete-circuit) ['([1 2 3 4 5] [3 4 5 1 2]) '([2 3 4] [3 4 3])])

;;137
(defn single-number [nums]
  (let [search (fn [num-map num]
                 (let [freq (or (get num-map num) 3)]
                   (if (= freq 1)
                     (dissoc num-map num)
                     (assoc num-map num (dec freq)))))]
    (first (keys (reduce search {} nums)))))
(map single-number [[2,2,3,2] [0,1,0,1,0,1,99]])

;;139
(defn word-break [s word-dict]
  (let [len-word-map (reduce (fn [dict word]
                               (let [len (count word)
                                     words (or (get dict len) #{})]
                                 (assoc dict len (conj words word))
                                 )) {} word-dict)
        lens (sort (keys len-word-map))]
    (letfn [(break [s]
              (cond
                (empty? s) true
                (< (count s) (first lens)) false
                :else (reduce (fn [result l]
                                (if (and (contains? (get len-word-map l) (subs s 0 l)) (break (subs s l)))
                                  (reduced true)
                                  result)) false lens)))]
      (break s))))
(defn word-break [s word-dict]
  (let [word-set (set word-dict)]
    (letfn [(break [s]
              (reduce (fn [result index]
                        (let [left (subs s 0 index)
                              right(subs s index (count s))]
                          (if (and (contains? word-set left)
                                   (break right))
                            (reduced true)
                            result)))
                      (empty? s)
                      (range 1 (inc (count s)))))]
      (break s))))
(map (partial apply word-break) ['("leetcode" ["leet" "code"]) '("applepenapple" ["apple" "pen"]) '("catsandog"  ["cats" "dog" "sand" "and" "cat"])])

;;150
(defn eval-RPN [tokens]
  (let [calculate (fn [exprs token]
                    (case token
                      "+" (+ (second exprs) (first exprs))
                      "-" (- (second exprs) (first exprs))
                      "*" (* (second exprs) (first exprs))
                      "/" (int (/ (second exprs) (first exprs)))
                      (Integer/parseInt token)))
        eval-expr (fn [exprs token]
                    (let [operator? (fn [token]
                                      (contains? #{"+" "-" "*" "/"} token))
                          value (calculate exprs token) ]
                      (if (operator? token)
                        (cons value (drop 2 exprs))
                        (cons value exprs))))]
   (first (reduce eval-expr (list) tokens))))
(map eval-RPN [["2" "1" "+" "3" "*"] ["4" "13" "5" "/" "+"] ["10" "6" "9" "3" "+" "-11" "*" "/" "*" "17" "+" "5" "+"]])

;;151
(defn reverse-words [s]
  (->> (str/split (str/trim s) #"\W+")
       reverse))

(defn reverse-words [s]
  (let [cs (vec (str/trim s))
        len (count cs)]
    (reduce (fn [[q word] index]
              (cond
                (= index len) (if (empty? word) [q []] [(cons (str/join "" word) q)])
                (not= (cs index) \ ) [q (conj word (cs index))]
                (not (empty? word)) [(cons (str/join "" word) q) []]
                :else [q word]))
            [[] []] (range (inc (count cs))))))
(map reverse-words ["the sky is blue" "  hello world  " "a good   example" "  Bob    Loves  Alice   " "Alice does not even like bob"])

;;152
(defn max-product [nums]
  (let [result (first nums)
        find-max-product (fn [[result max-value min-value] num]
                           (let [[max-value min-value] (if (neg? num)
                                                         [min-value max-value]
                                                         [max-value min-value])
                                 max-value' (max num (* max-value num))
                                 min-value' (min num (* min-value num))
                                 result' (max max-value' result)]
                             [result' max-value' min-value']))]
    (first (reduce find-max-product [result result result] (rest nums)))))
(defn max-product [nums]
  (let [len (count nums)
        nums' (vec (reverse nums))
        get-product (fn [nums xs index]
                      (let [num (if (zero? (xs (dec index)))
                                  1
                                  (xs (dec index)))]
                        (* (nums index) num)))
        add-product (fn [nums xs index]
                      (conj xs (get-product nums xs index)))]
    (->> (reduce (fn [[xs ys] index]
              [(add-product nums xs index)
               (add-product nums' ys index)]
                   ) [[(first nums)] [(first nums')]] (range 1 len))
         (flatten)
         (apply max))))
(map max-product [[2 3 -2 4] [-2 0 -1]])
