(ns leetcode.core
  (:gen-class)
  (:require [clojure.string :as str]
            [clojure.set :as set]))

;;118
(defn generate [num-rows]
  (reduce (fn [results index]
            (let [row (last results)
                  result (mapv (partial +) (cons 0 row) (conj row 0))]
              (conj results result)))
          [[1]] (range 1 num-rows)))
(map generate [5 1])

;;119
(defn get-row [row-index]
  (let [generate-row (fn [result _]
                       (mapv (partial +) (cons 0 result) (conj result 0)))]
  (reduce generate-row [1] (range row-index))))
(map get-row [3 0 1])

;;594
(defn find-LHS [nums]
  (let [freqs (frequencies nums)
        numbers (keys freqs)]
    (reduce (fn [max-length num]
              (let [cnt1 (get freqs num)
                    cnt2 (get freqs (inc num))]
                (if (nil? cnt2)
                  max-length
                  (max max-length (+ cnt1 cnt2))))) 0 numbers)))
(map find-LHS [[1 3 2 2 5 2 3 7] [1 2 3 4] [1 1 1 1]])

;;598
(defn max-count [m n ops]
  (let [min-col (reduce min n (map last ops))
        min-row (reduce min m (map first ops))]
    (* min-row min-col)))
(map (partial apply max-count) ['(3 3 [[2 2] [3 3]]) '(3 3 [[2 2] [3 3] [3 3] [3 3] [2 2] [3 3] [3 3] [3 3] [2 2] [3 3] [3 3] [3 3]]) '(3 3 [])])

;;680
(defn valid-palindrome [s]
  (letfn [(valid? [cs left right deleted]
            (loop [l left r right deleted deleted]
              (cond
                (>= l r) true
                (and (not= (cs l) (cs r)) deleted) false
                (and (not= (cs l) (cs r)) (not deleted)) (or (valid? cs (inc l) r true) (valid? cs l (dec r) true))
                :else (recur (inc l) (dec r) deleted))))]
    (if (= s (str/reverse s))
      true
      (valid? (vec s) 0 (dec (count s)) false))))
(map valid-palindrome ["aba" "abca" "abc"])

;;733
(defn flood-fill [image sr sc new-color]
  (let [row (count image)
        col (count (image 0))
        matrix (make-array Long/TYPE row col)
        indexes (for [r (range row) c (range col)]
                  [r c])
        clone #(doseq [[r c] indexes]
                 (aset matrix r c ((image r) c)))
        start-color ((image sr) sc)
        valid? (fn [[r c]]
                 (and (>= r 0) (< r row) (>= c 0) (< c col) (= (aget matrix r c) start-color)))]
    (letfn [(fill [[r c]]
              (when (valid? [r c])
                (aset matrix r c new-color)
                (doseq [[r' c'] [[(dec r) c] [(inc r) c] [r (dec c)] [r (inc c)]]]
                  (fill [r' c']))))]
      (clone)
      (fill [sr sc]))
      (mapv vec matrix)))
(map (partial apply flood-fill) ['([[1,1,1],[1,1,0],[1,0,1]] 1 1 2) '([[0,0,0],[0,0,0]] 0 0 2)])

;;821
(defn shortest-to-char [s c]
  (let [len (count s)
        cs (vec s)
        c (first (vec c))
        count-distance (fn [cs [distances start] index]
                         (let [distance (if (= (cs index) c)
                                          0
                                          (- index start))
                               start' (if (= (cs index) c)
                                        index
                                        start)]
                           [(conj distances distance) start']))
        get-distances (fn [cs] (->> (range len)
                                    (reduce (partial count-distance cs) [[] (- len)])
                                    (first)))
        distances1 (get-distances cs)
        distances2 (reverse (get-distances (vec (reverse cs))))]
    (map min distances1 distances2)))
(map (partial apply shortest-to-char) ['("loveleetcode" "e") '("aaab" "b")])

;;836
(defn is-rectangle-overlap [rec1 rec2]
  (let [[x1 y1 x2 y2] rec1
        [x1' y1' x2' y2'] rec2]
    (and (> (min x2 x2') (max x1  x1'))
         (> (min y2 y2') (max y1 y1')))))
(map (partial apply is-rectangle-overlap) ['([0 0 2 2] [1 1 3 3]) '([0 0 1 1] [1 0 2 1]) '([0 0 1 1] [2 2 3 3])])

;;883
(defn projection-area [grid]
  (let [sum (fn [xs] (apply + xs))
        transpose (fn [m]
                    (let [row (count m)
                          col (count (m 0))
                          mt (make-array (type ((m 0) 0)) col row)]
                      (doseq [r (range row) c (range col)]
                        (aset mt c r ((m r) c)))
                      (mapv vec mt)))
        count-cells #(reduce (fn [sum n]
                               (if (pos? n) (inc sum) sum)) 0 %)
        top (sum (map count-cells grid))
        front (sum (map #(apply max %) grid))
        side (sum (map #(apply max %) (transpose grid)))]
    (+ top front side)))
(map projection-area [[[1 2] [3 4]] [[2]] [[1 0] [0 2]] [[1 1 1] [1 0 1] [1 1 1]] [[2 2 2] [2 1 2] [2 2 2]]])

;;888
(defn fair-candy-swap [alice-sizes bob-sizes]
  (let [s1 (apply + alice-sizes)
        s2 (apply + bob-sizes)
        delta (/ (- s1 s2) 2)
        alice-set (set alice-sizes)
        bob-set (set bob-sizes)
        get-fair-sizes (fn [sizes size]
                (let [size2 (- size delta)]
                  (if (contains? bob-set size2)
                    (reduced [size size2])
                    sizes)))]
    (reduce get-fair-sizes [] alice-set)))
(map (partial apply fair-candy-swap) ['([1 1] [2 2]) '([1 2] [2 3]) '([2] [1 3]) '([1 2 5] [2 4])])

;;925
(defn is-long-pressed-name1 [name typed]
  (let [group-string (fn [s]
                       (let [len (count s)
                             cs (vec s)
                             split-string' (fn [[results result] index]
                                             (cond
                                               (= index len) [(conj results result) []]
                                               (= (cs (dec index)) (cs index)) [results (conj result (cs index))]
                                               :else [(conj results result) [(cs index)]]))]
                         (->> (range 1 (inc len))
                              (reduce split-string' [[] [(first cs)]])
                              (first)
                              (map #(str/join "" %)))))
        words1 (group-string name)
        words2 (group-string typed)]
    (if (not= (count words1) (count words2))
      false
      (->> (map str/includes?  words2 words1)
           (every? true?)))))

(defn is-long-pressed-name [name typed]
  (let [len1 (count name)
        cs1 (vec name)
        cs2 (vec typed)
        compare-char (fn [index1 index2]
                       (cond
                         (and (< index1 len1) (= (cs1 index1) (cs2 index2))) (inc index1)
                         (or (zero? index1) (not= (cs2 (dec index2)) (cs2 index2))) (reduced index1)
                         :else index1))
        last-index (reduce compare-char 0 (range (count typed)))]
    (= last-index len1)))
(map (partial apply is-long-pressed-name) ['("alex" "aaleex") '("saeed" "ssaaedd") '("leelee" "lleeelee") '("laiden" "laiden")])

;;937
(defn reorder-log-files [logs]
  (let [get-log-content #(subs % (inc (str/index-of % \ )))
        compare-content (fn [l r]
                          (compare (get-log-content l) (get-log-content r)))
        digit-log? #(Character/isDigit (last (vec %)))
        letter-log? (comp not digit-log?)
        letter-logs (sort compare-content (filter letter-log? logs))
        digit-logs (filter digit-log? logs)]
    (concat letter-logs digit-logs)))
(map reorder-log-files [["dig1 8 1 5 1" "let1 art can" "dig2 3 6" "let2 own kit dig" "let3 art zero"] ["a1 9 2 3 1" "g1 act car" "zo4 4 7" "ab1 off key dog" "a8 act zoo"]])

;; (defn valid-mountain-array [arr]
;;   (let [increasing? (fn [coll]
;;                       (apply < coll))
;;         len (count arr)
;;         indexes (range 1 (dec len))
;;         valid? (fn [[result peak-counter] index]
;;                  (let [x0 (arr (dec index))
;;                        x1 (arr index)
;;                        x2 (arr (inc index))]
;;                    (cond
;;                      (and (>= x0 x1) (> x1 x2) (< x0 x1)) (if (zero? peak-counter)
;;                                                             [result (inc peak-counter)]
;;                                                             (reduced [false (inc peak-counter)]))
;;                      (< x0 x1 x2) [result peak-counter]
;;                      :else (reduced [false peak-counter]))))]
;;     (if (or (increasing? (rest arr)) (increasing? (drop-last arr)))
;;       true
;;       (first (reduce valid? [true 0] indexes)))))
;; ;;941
(defn valid-mountain-array1 [arr]
  (let [valid? (fn [coll]
                 (let [max-value (apply max coll)
                       index (.indexOf coll max-value)]
                   (and (apply < (subvec coll 0 (inc index)))
                        (apply > (subvec coll index (count coll))))))]
        (if  (< (count arr) 3)
          false
          (valid? arr))))
(defn valid-mountain-array [arr]
  (let [max-value-index1 (reduce (fn [result index]
                            (if (< (arr (dec index)) (arr index))
                              result
                              (reduced (dec index)))) nil (range 1 (count arr)))
        max-value-index2 (reduce (fn [result index]
                              (if (> (arr (dec index)) (arr index))
                                    result
                                    (reduced index)))
                                  nil
                              (range (dec (count arr)) 0 -1))]
    (and (> (count arr) 2) (not (nil? max-value-index1))
         (= max-value-index1 max-value-index2))))
(map valid-mountain-array [[2 1] [3 5 5] [0 3 2 1]])

;;942
(defn di-string-match [s]
  (let [len (count s)
        cs (vec s)
        nums (vec (range 0 (+ len 1)))
        append (fn [[result left right] index]
                 (let [c (cs index)]
                   (cond
                     (zero? index) (if (= c \I)
                                     [[(nums left) (nums right)] (inc left) (dec right)]
                                     [[(nums right) (nums left)] (inc left) (dec right)])
                     (= \c \I) [(conj result (nums right)) left (dec right)]
                     :else [(conj result (nums left)) (inc left) right])))]
   (first (reduce append [[] 0 len] (range len)))))
(map di-string-match ["IDID" "III" "DDI"])

;;977
(defn sorted-squares [nums]
  (let [abs (fn [n] (if (neg? n) (- n) n))
        square (fn [n] (* n n))
        len (count nums)
        append (fn [[result index1 index2] _]
                 (let [num1 (abs (nums index1))
                       num2 (abs (nums index2))]
                           (if (> num1 num2)
                             [(cons (square num1) result) (inc index1) index2]
                             [(cons (square num2) result) index1 (dec index2)])))]
   (first (reduce append [[] 0 (dec len)] (range (count nums))))))
(map sorted-squares [[-4,-1,0,3,10] [-7,-3,2,3,11]])

;;997
(defn find-judge [n trust]
  (let [trust-graph (make-array Long/TYPE (inc n) 2)
        judge? (fn [index]
                 (and (= (aget trust-graph index 1) (dec n))
                      (zero? (aget trust-graph index 0))))]
    (doseq [[person trustee] trust]
      (aset trust-graph person 0 (inc (aget trust-graph person 0)))
      (aset trust-graph trustee 1 (inc (aget trust-graph trustee 1))))
    (->> (filter judge? (range 1 (inc n)))
         (#(if (= (count %) 1) (first %) -1)))))
(map (partial apply find-judge) ['(2 [[1 2]]) '(3 [[1 3] [2 3]]) '(3 [[1 3] [2 3] [3 1]]) '(3 [[1 2] [2 3]]) '(4 [[1 3] [1 4] [2 3] [2 4] [4 3]])])

;;999
(defn num-rook-captures [board]
  (let [row (count board)
        col (count (board 0))
        indexes (for [r (range row) c (range col)]
                  [r c])
        find-rook (fn [result [r c]]
                    (if (= ((board r) c) "R")
                      (reduced [r c])
                      result))
        [start-r start-c] (reduce find-rook [] indexes)
        deltas [[-1 0] [1 0] [0 1] [0 -1]]
        valid? (fn [r c]
                 (not (or (< r 0) (< c 0) (>= r row) (>= c col) (= ((board r) c) "B"))))
        capture (fn [r c [delta-r delta-c]]
                  (loop [r (+ r delta-r) c (+ c delta-c)]
                    (if (valid? r c)
                      (if (= ((board r) c) "p")
                        1
                        (recur (+ r delta-r) (+ c delta-c)))
                      0)))
        capture-all (fn [r c]
                      (apply + (map #(capture r c %) deltas)))]
    (capture-all start-r start-c)
    ))
(map num-rook-captures [[["." "." "." "." "." "." "." "."] ["." "." "." "p" "." "." "." "."] ["." "." "." "R" "."   "." "." "p"] ["." "." "." "." "." "." "." "."] ["." "." "." "." "." "." "." "."] ["." "." "." "p" "." "." "." "."] [". " "." "." "." "." "." "." "."] ["." "." "." "." "." "." "." "."]] [["." "." "." "." "." "." "." "."] ["." "p" "p" "p" "p" "p" "." "."] ["." "p" "p" "B"     "p" "p" "." "."] ["." "p" "B" "R" "B" "p" "." "."] ["." "p" "p" "B" "p" "p" "." "."] ["." "p" "p" "p" "p" "p" "." ".   "] ["." "." "." "." "." "." "." "."] ["." "." "." "." "." "." "." "."]] [["." "." "." "." "." "." "." "."] ["." "." "." "p" "." "." "." "."] ["." "." "." "p" ".  " "." "." "."] ["p" "p" "." "R" "." "p" "B" "."] ["." "." "." "." "." "." "." "."] ["." "." "." "B" "." "." "." "."]   ["." "." "." "p" "." "." "." "."] ["." "." "." "." "." "." "." "."]]])

;;1021
(defn remove-outer-parentheses1 [s]
  (let [cs (vec s)
        len (count s)
        append (fn [[results result lefts] index]
                 (if (= index len)
                   [(conj results result) [] 0]
                   (let [c (cs index)]
                     (cond
                       (and (= c \)) (> lefts 1)) [results (conj result c) (dec lefts)]
                       (and (= c \)) (= lefts 1)) [(conj results (conj result c)) [] (dec lefts)]
                       :else [results (conj result c) (inc lefts)]))))]
    (->> (reduce append [[] [] 0] (range (inc len)))
         (first)
         (map #(drop-last (rest %)))
         (flatten)
         (str/join "")
         )))

(defn remove-outer-parentheses [s]
  (let [cs (vec s)
        append (fn [[result lefts] index]
                 (let [c (cs index)]
                   (cond
                     (and (= c \() (pos? lefts)) [(conj result c) (inc lefts)]
                     (and (= c \() (zero? lefts)) [result (inc lefts)]
                     (and (= c \)) (> lefts 1)) [(conj result c) (dec lefts)]
                     :else [result (dec lefts)])))]
    (->> (reduce append [[] 0] (range (count s)))
         (first)
         (str/join ""))))
(map remove-outer-parentheses ["(()())(())" "(()())(())(()(()))" "()()"])

;;1046
(defn last-stone-weight [stones]
  (letfn [(play [stones]
            (let [stones (vec (sort > stones))
                  y (first stones)
                  x (second stones)]
              (cond
                (empty? stones) 0
                (= (count stones) 1) y
                (= y x) (play (drop 2 stones))
                :else (play (conj (drop 2 stones) (- y x))))))]
    (play stones)))
(map last-stone-weight [[2 7 4 1 8 1] [1]])

;;1047
(defn remove-duplicates1 [s]
  (letfn [(remove-duplicates' [cs]
            (let [len (count cs)
                  append (fn [[result start] index]
                           (cond
                             (= (cs start) (cs index)) [result start]
                             (= (+ start 2) len) [(vec (concat result (subvec cs start len))) len]
                             (= (inc index) len) [(conj result (cs index)) len]
                             (= (inc start) index) [(conj result (cs start)) index]
                             :else [result index]))
                  result (->> (reduce append [[] 0] (range 1 len))
                              (first))]
              (println result)
              (cond
                (<= (count cs) 1) cs
                (= result cs) result
                :else (remove-duplicates' result))))]
    (str/join "" (remove-duplicates' (vec s)))))

(defn remove-duplicates [s]
  (let [append (fn [result c]
                 (if (and (not-empty result) (= c (last result)))
                   (vec (drop-last result))
                   (conj result c)))]
  (reduce append [] (vec s))))
(map remove-duplicates ["abbaca" "azxxzy"])

;;1071
(defn gcd-of-strings1 [str1 str2]
  (letfn [(gcd [a b]
            (if (zero? (rem a b))
              b
              (gcd b (rem a b))))]
    (let [divisor (gcd (count str1) (count str2))
          s1 (str str1 str2)
          common-str-divisor (subs str1 0 divisor)
          s2 (->> common-str-divisor
                  (#(take (quot (count s1) divisor) (cycle [%])))
                  (str/join ""))]
      (if (= s1 s2)
        common-str-divisor
        ""))))

(defn gcd-of-strings2 [str1 str2]
  (letfn [(gcd [a b]
             (if (zero? (rem a b))
               b
               (gcd b (rem a b))))]
    (if (= (str str1 str2) (str str2 str1))
      (subs str1 0 (gcd (count str1) (count str2)))
      "")))
(defn gcd-of-strings [str1 str2]
  (letfn [(gcd [s1 s2]
            (cond
              (< (count s1) (count s2)) (gcd s2 s1)
              (not (str/starts-with? s1 s2)) ""
              (= s2 "") s1
              :else (gcd s2 (subs s1 (count s2)))))]
    (gcd str1 str2)))
(map (partial apply gcd-of-strings) ['("ABCABC" "ABC") '("ABABAB" "ABAB") '("LEET" "CODE") '("ABCDEF" "ABC")])

;;1089
(defn duplicate-zeros1 [arr]
  (letfn [(get-index [arr left right]
            (cond
              (= left right) left
              (= (arr left) 0) (get-index arr (inc left) (- right 1))
              :else (get-index arr (inc left) right)))
          (duplicate [matrix left right]
            (cond
              (< left 0) (vec matrix)
              (= (aget matrix left) 0) (do (aset matrix right 0)
                                           (aset matrix (dec right) 0)
                                           (duplicate matrix (dec left) (- right 2)))
              :else (do (aset matrix right (aget matrix left))
                  (duplicate matrix (dec left) (dec right)))))]
    (let [matrix (into-array arr)
          left (get-index arr 0 (dec (count arr)))]
      (duplicate matrix left (dec (count arr))))))
(defn duplicate-zeros [arr]
  (let [len (count arr)
        indexes (range (dec len) -1 -1)
        matrix (into-array arr)
        zeroes (count (filter zero? arr))
        duplicate (fn [zeroes index]
                    (let [index' (+ index zeroes)
                          digit (aget matrix index)]
                      (cond
                        (zero? digit) (do
                                        (when (< index' len)
                                          (aset matrix index' 0))
                                        (when (< (dec index') len)
                                          (aset matrix (dec index') 0))
                                        (dec zeroes))
                        (and (< index' len)
                             (not= digit 0)) (do
                                               (aset matrix index' (aget matrix index))
                                               zeroes)
                        :else zeroes)))]
    (reduce duplicate zeroes indexes)
    (vec matrix)))
(map duplicate-zeros [[1 0 2 3 0 4 5 0] [1 2 3]])

;;1103
(defn distribute-candies [candies num-people]
  (let [candy-list (make-array Long/TYPE num-people)
        distribute (fn [candies num-people]
                     (loop [given 1 candies candies]
                       (let [index (rem (dec given) num-people)
                             given' (min given candies)]
                         (aset candy-list index (+ (aget candy-list index) given'))
                         (when (pos? (- candies given'))
                           (recur (inc given) (- candies given'))))))]
    (distribute candies num-people)
    (vec candy-list)))
(map (partial apply distribute-candies) ['(7 4) '(10 3)])

;;1175
(defn num-prime-arrangements [n]
  (let [prime?' (fn [n]
                  (reduce #(if (zero? (rem n %2))
                             (reduced false)
                             %1) true (range 2 (inc (int (Math/sqrt n))))))
        prime? (fn [n] (case n
                         1 false
                         2 true
                         (prime?' n)))
        count-primes (fn [n]
                       (count (filter prime? (range 2 (inc n)))))
        num-primes (count-primes n)
        modulo (fn [n]
                 (rem n 1000000007))
        factorial (fn [n]
                    (reduce #(modulo (* %2 %1)) 1 (range 1 (inc n))))]
    (modulo (* (factorial num-primes) (factorial (- n num-primes))))))
(map num-prime-arrangements [5 100])

;;1184
(defn distance-between-bus-stops [distance start destination]
  (let [len (count distance)
        get-distance (fn [start end]
                       (let [sum-distance (fn [indexes]
                                  (apply + (map #(distance (rem % len)) indexes)))]
                       (if (< start end)
                         (sum-distance (range start end))
                         (sum-distance (range end (+ start len))))))]
    (min (get-distance start destination)
         (get-distance destination start))))
(map (partial apply distance-between-bus-stops) ['([1,2,3,4] 0 1) '([1 2 3 4] 0 2) '([1 2 3 4] 0 3)])


(defn make-fancy-string [s]
  (let [cs (vec s)
        len (count s)
        append-fancy-string (fn [[result start] index]
                              (cond
                                (= index len) [(str result (subs s start (+ start (min 2 (- index start))))) len]
                                (= (cs (dec index)) (cs index)) [result start]
                                :else [(str result (subs s start (+ start (min 2 (- index start))))) index]))
        indexes (range 1 (inc len))]
   (first (reduce append-fancy-string ["" 0] indexes))))
(map make-fancy-string ["leeetcode" "aaabaaaa" "aab"])

;;1217
(defn min-cost-to-move-chips [position]
  (let [count-chip (fn [[evens odds] p]
              (if (even? p)
                [(inc evens) odds]
                [evens (inc odds)]))]
    (->> (reduce count-chip [0 0] position)
         (apply min))))
(map min-cost-to-move-chips [[1 2 3] [2 2 2 3 3] [1 1000000000]])

;;1221
(defn balanced-string-split1 [s]
  (let [count-string (fn [[num-strs pairs] c]
                       (cond
                         (and (= c \L) (= pairs -1)) [(inc num-strs) 0]
                         (and (= c \R) (= pairs 1)) [(inc num-strs) 0]
                         (= c \L) [num-strs (inc pairs)]
                         :else [num-strs (dec pairs)]))]
    (first (reduce count-string [0 0] (vec s)))))

(defn balanced-string-split [s]
  (let [count-string (fn [[num-strs pairs] c]
                       (let [pairs (if (= c \L) (inc pairs) (dec pairs))
                             num-strs (if (zero? pairs) (inc num-strs) num-strs)]
                         [num-strs pairs]))]
    (first (reduce count-string [0 0] (vec s)))))
(map balanced-string-split ["RLRRLLRLRL" "RLLLLRRRLR" "LLLLRRRR" "RLRRRLLRLL"])

;;1232
(defn check-straight-line [coordinates]
  (let [to-vector (fn [index]
                    (let [[x1 y1] (coordinates (dec index))
                          [x2 y2] (coordinates index)]
                      [(- x1 x2) (- y1 y2)]))
        straight-line? (fn [[u1 u2] [v1 v2]]
                         (= (* u1 v2) (* u2 v1)))
        vectors (reduce #(conj %1 (to-vector %2)) [] (range 1 (count coordinates)))
        check #(if (straight-line? (vectors (dec %2)) (vectors %2))
                 %1
                 (reduced false))]
    (reduce check true (range 1 (count vectors)))))
(map check-straight-line [[[1 2] [2 3] [3 4] [4 5] [5 6] [6 7]] [[1 1] [2 2] [3 4] [4 5] [5 6] [7 7]]])

;;1252
(defn odd-cells [m n indices]
  (let [rows (make-array Long/TYPE m)
        cols (make-array Long/TYPE n)
        count-odds (fn [sum [r c]]
                     (let [value (+ (aget rows r) (aget cols c))]
                       (if (odd? value)
                         (inc sum)
                         sum)))]
    (doseq [[r c] indices]
      (aset rows r (inc (aget rows r)))
      (aset cols c (inc (aget cols c))))
    (->> (for [r (range m) c (range n)]
           [r c])
         (reduce count-odds 0))))
(map (partial apply odd-cells) ['(2 3 [[0 1] [1 1]]) '(2 2 [[1 1] [0 0]])])

(apply mapv vector [[1 2] [3 4]])

;;1275
(defn tictactoe [moves]
  (let [matrix (make-array Character/TYPE 3 3)
        transpose #(apply mapv vector %)
        any-row-win? (fn [matrix player]
                       (reduce (fn [result index]
                                 (if (= (set (get matrix index)) #{player})
                                   (reduced true)
                                   result))
                               false (range 3)))
        any-column-win? (fn [matrix player]
                          (let [m (transpose matrix)]
                            (any-row-win? m player)))
        any-diagonal-win? (fn [matrix player]
                            (let [d1 (map #(aget matrix % %) (range 3))
                                  d2 (map #(aget matrix (- 2 %) %) (range 3))]
                              (or (apply = (conj d1 player)) (apply = (conj d2 player)))))
        judge (fn [matrix player]
                (if (or (any-column-win? matrix player)
                     (any-row-win? matrix player)
                     (any-diagonal-win? matrix player)) player
                 nil))
        play (fn [result index]
               (let [[r c] (moves index)
                     player (if (even? index)
                              \X
                              \O)]
                 (aset matrix r c player)
                 (judge matrix player)))
        no-more-moves? (fn [matrix]
                         (->>(map #(filter (fn [c] (and (not= c \X) (not= c \O))) %) matrix)
                         (flatten)
                         (#(zero? (count %)))))
    winner (reduce play nil (range (count moves)))]
    (cond
      (= winner \X) "A"
      (= winner \O) "B"
      (no-more-moves? matrix) "Draw"
      :else "Pending")))
(map tictactoe [[[0 0] [2 0] [1 1] [2 1] [2 2]] [[0 0] [1 1] [0 1] [0 2] [1 0] [2 0]] [[0 0] [1 1] [2 0] [1 0] [1 2] [2 1] [0 1] [0 2] [2 2]]
                [[0 0] [1 1]]])

;;1309
(defn freq-alphabets [s]
  (letfn [(->letter [num]
            (char (+ (int \a) (dec (Integer/parseInt num)))))
          (decode-single-digit [s index]
            (->letter (subs s (- index 2) index)))
          (decode-double-digits [s index]
            (->letter (subs s index (inc index))))
          (double-digits? [s index]
            (= (subs s index (inc index)) "#"))
          (decode [s index]
            (cond
              (neg? index) ""
              (double-digits? s index) (str (decode s (- index 3)) (decode-single-digit s index))
              :else (str (decode s (- index 1)) (decode-double-digits s index))))]
    (decode s (dec (count s)))))
(map freq-alphabets ["10#11#12" "1326#" "25#" "12345678910#11#12#13#14#15#16#17#18#19#20#21#22#23#24#25#26#"])

;;1370
(defn sort-string [s]
  (let [freqs (frequencies (vec s))]
    (letfn [(arrange [freqs increase]
              (let [letters (let [compare-letter (if increase
                                                   #(compare %1 %2)
                                                   #(compare %2 %1))]
                              (sort compare-letter (keys freqs)))
                    decrease-letter-count (fn [m letter]
                                            (let [cnt (get m letter)]
                                              (if (= cnt 1)
                                                (dissoc m letter)
                                                (assoc m letter (dec cnt)))))
                    freqs' (reduce decrease-letter-count freqs letters)]
                (if (empty? letters)
                  ""
                  (str (str/join "" letters) (arrange freqs' (not increase))))))]
      (arrange freqs true))))
(map sort-string ["aaaabbbbcccc" "rat" "leetcode" "ggggggg" "spo"])

;;1385
(defn find-the-distance-value [arr1 arr2 d]
  (let [abs (fn [n]
              (if (neg? n)
                (- n)
                n))
        distance-element? (fn [x]
                            (every? #(> (abs (- x %)) d) arr2))]
       (count (filter distance-element? arr1))))
(map (partial apply find-the-distance-value) ['([4 5 8] [10 9 1 8] 2) '([1 4 2 3] [-4 -3 6 10 20 30] 3) '([2 1 100 3] [-5 -2 10 -3 7] 6)])

;;1380
(defn lucky-numbers [matrix]
  (let [mt (apply map vector matrix)
        row-maxs (map #(apply min %) matrix)
        col-mins (map #(apply max %) mt)
        max-indexes (map (fn [row value index] [index (.indexOf row value)]) matrix row-maxs (range (count matrix)))
        min-indexes (map (fn [col value index] [(.indexOf col value) index]) mt col-mins (range (count mt)))]
    (->> (set/intersection (set max-indexes) (set min-indexes))
         (map #((matrix (first %)) (last %))))))
(map lucky-numbers [[[3 7 8] [9 11 13] [15 16 17]] [[1 10 4 2] [9 3 8 7] [15 16 17 12]] [[7 8] [1 2]] [[3 6] [7 1] [5 2] [4 8]]])

;;1399
(defn count-largest-group [n]
  (let [sum-digits (fn [n]
                     (->> (str n)
                          (map #(- (int %) (int \0)))
                          (apply +)))]
    (->> (range 1 (inc n))
         (map sum-digits)
         (frequencies)
         (vals)
         (#(filter (fn [x] (= x (apply max %))) %))
         (count))))
(map count-largest-group [13 2 15 24])

;;1389
(defn create-target-array [nums index]
  (let [insert (fn [result i]
                 (let [index' (index i)
                       num (nums i)]
                (vec (concat (conj (subvec result 0 index') num) (subvec result index')))))]
  (reduce insert [] (range (count nums)))))
(map (partial apply create-target-array) ['([0 1 2 3 4] [0 1 2 2 1]) '([1 2 3 4 0]  [0 1 2 3 0]) '([1] [0])])

;;1417
(defn reformat [s]
  (let [reformatable? (fn [s]
                        (let [cs (vec s)
                              num-letters (count (filter #(Character/isLetter %) cs))
                              num-digits (- (count s) num-letters)
                              delta (- num-letters num-digits)]
                          (contains? #{-1 0 1} delta)))
        reformat' (fn [s]
                    (let [cs (vec s)
                          letters (filterv #(Character/isLetter %) cs)
                          digits (filterv #(Character/isDigit %) cs)
                          [long-chars short-chars] (if (> (count letters) (count digits))
                                                     [letters digits]
                                                     [digits letters])]
                      (conj (reduce (fn [result index]
                                      (conj result (long-chars index) (short-chars index))) [] (range (count short-chars))) (last long-chars))))]
    (if (reformatable? s)
      (reformat' s)
      "")))
(map reformat ["a0b1c2" "leetcode" "1229857369" "covid2019" "ab123"])

;;1475
(defn final-prices1 [prices]
  (let [len (count prices)
        indexes (range (- len 1) -1 -1)
        get-final-price (fn [index]
                          (let [price (prices index)
                                get-final-price' (fn [result discount]
                                                   (if (>= price discount)
                                                     (reduced (- price discount))
                                                              result))]
                            (reduce get-final-price' price (subvec prices (inc index)))))]
   (reverse (map get-final-price indexes))))
(defn final-prices [prices]
  (let [len (count prices)
        price-list (into-array prices)
        get-final-price (fn [stack index]
                          (loop [stack stack]
                            (if (and (not-empty stack) (>= (aget price-list (last stack)) (aget price-list index)))
                              (do
                                (aset price-list (last stack) (- (aget price-list (last stack)) (aget price-list index)))
                                (recur (pop stack)))
                              stack)))
        get-final-price (fn [stack index]
                          (conj (get-final-price stack index) index))]
    (reduce get-final-price [] (range len))
   ; (vec price-list)
    ))
(map final-prices [[8 4 6 2 3] [1 2 3 4 5] [10 1 1 6]])

;;1539
(defn find-kth-positive [arr k]
  (let [num-set (set arr)
        find-kth (fn [[result k] n]
                   (cond
                     (contains? num-set n) [result k]
                     (= k 1) (reduced [n 0])
                     :else [result (dec k)]))]
    (first (reduce find-kth [nil k] (range 1 1001)))))
(map (partial apply find-kth-positive) ['([2 3 4 7 11] 5) '([1 2 3 4] 2)])

;;1556
(defn thousand-separator [n]
  (letfn [(add-separator [n]
            (let [q (quot n 1000)
                  remainder (rem n 1000)]
              (if (zero? q)
                (str remainder)
                (str (add-separator q) "." remainder))))]
      (add-separator n)))
(defn thousand-separator [n]
  (let [digits (vec (str n))
        len (count digits)
        add-separator (fn [result index]
                        (if (and (pos? index) (zero? (rem (- len index) 3)))
                          (str result "." (digits index))
                          (str result (digits index))))]
    (reduce add-separator "" (range len))))
(map thousand-separator [987 1234 123456789 0])

;;1544
(defn make-good [s]
  (let [bad? (fn [c1 c2]
               (and (not= c1 c2) (= (Character/toUpperCase c1) (Character/toUpperCase c2))))
        remove-bad-chars (fn [result c]
                           (cond
                             (empty? result) [c]
                             (bad? (last result) c) (pop result)
                             :else (conj result c)))]
    (str/join "" (reduce remove-bad-chars [] (vec s)))))
(map make-good ["leEeetcode" "abBAcC" "s"])

;;1560
(defn most-visited [n rounds]
  (let [get-range (fn [rounds start]
                    (let [append (fn [result index] (if (< (rounds (dec index)) (rounds index))
                                                      (conj result (rounds index))
                                                      (reduced result)))]
                      (-> (reduce append [(first rounds)] (range 1 (count rounds)))
                          (sort)
                          (#(range start (inc (last %)))))))
        r1 (get-range rounds (first rounds))
        r2 (get-range (vec (reverse rounds)) 1)
        most-visited' (fn []
                        (->> (set/intersection (set r1) (set r2))
                        (vec)
                        (sort)
                        ))]
    (if (= (sort rounds) rounds)
      (range (first rounds) (inc (last rounds)))
      (most-visited'))))
(map (partial apply most-visited) ['(4 [1 3 1 2]) '(2 [2 1 2 1 2 1 2 1 2]) '(7 [1 3 5 7])])

;;1566
(defn contains-pattern [arr m k]
  (let [len (count arr)
        compare-pattern (fn [index]
                          (and (= (arr index) (arr (+ index m)))
                               (= (subvec arr index (+ index m)) (subvec arr (+ index m) (+ index m m)))))]
    (letfn [(search-pattern [matched-len index]
              (cond
                (> index (- len m m)) false
                (compare-pattern index) (if (= (inc matched-len) (dec k))
                                          true
                                          (search-pattern (inc matched-len) (+ index m)))
                :else (search-pattern 0 (inc index))))]
      (search-pattern 0 0))))
(map (partial apply contains-pattern) ['([1 2 4 4 4 4] 1 3) '([1 2 1 2 1 1 1 3] 2 2) '([1 2 1 2 1 3] 2 3) '([1 2 3 1 2] 2 2) '([2 2 2 2] 2 3)])

;;1588
(defn sum-odd-length-subarrays [arr]
  (let [len (count arr)
        sum-subarrays (fn [size]
                        (let [indexes (range size (- (inc len) size))
                              sum (apply + (subvec arr 0 size))
                              add-sum (fn [[result sum] index]
                                        (let [sum' (- (+ sum (arr index)) (arr (- index size)))]
                                          [(+ result sum') sum']))]
                          (first (reduce add-sum [sum sum] indexes))))]

    (->> (map sum-subarrays (range 1 (inc len) 2))
         (apply +))))
(map sum-odd-length-subarrays [[1 4 2 5 3] [1 2] [10 11 12]])

;;1576
(defn modify-string [s]
  (let [letters #{"a" "b" "c"}
        cs (into-array (map str (vec s)))
        get-letter #(aget cs %)
        len (count s)
        replace-question-mark (fn [index]
                                (let [left (if (pos? index) (get-letter (dec index)) "")
                                      right (if (< index (dec len)) (get-letter (inc index)) "")
                                      middle (get-letter index)]
                                  (if (= middle "?")
                                    (aset cs index (first (disj (disj letters left) right)))
                                    middle)))]
    (doseq [index (range len)]
      (replace-question-mark index))
    (str/join "" cs)))
(map modify-string ["?zs" "ubv?w" "j?qg??b" "??yw?ipkj?"])

;;1582
(defn num-special [mat]
  (let [mt (apply mapv vector mat)
        special? (fn [index]
               (let [xs (mat index)]
                 (and (= 1 (apply + xs))
                      (= 1 (apply + (mt (.indexOf xs 1)))))))]
        (count (filter special? (range (count mat))))))
(map num-special [[[1 0 0] [0 0 1] [1 0 0]] [[1 0 0] [0 1 0] [0 0 1]] [[0 0 0 1] [1 0 0 0] [0 1 1 0] [0 0 0 0]] [[0 0 0 0 0] [1 0 0 0 0] [0 1 0 0 0] [0 0 1 0 0] [0 0 0 1 1]]])

;;1608
(defn special-array [nums]
  (let [nums (vec (sort nums))
        len (count nums)
        find-special (fn [result index]
                       (if (and (>= (nums index) (- len index))
                                (or (= index 0)
                                    (< (nums (dec index)) (- len index))))
                         (reduced (- len index))
                         result))]
    (reduce find-special -1 (range len))))
(map special-array [[3 5] [0 0] [0 4 3 0 4] [3 6 7 7 0]])

;;1629
(defn slowest-key [release-times key-pressed]
  (let [len (count release-times)
        cs (vec key-pressed)
        durations (reduce #(conj %1 (- %2 (last %1))) [(first release-times)] release-times)
        key-duration-map (reduce (fn [m index]
                                   (let [key (cs index)
                                         duration (durations index)]
                                     (assoc m key (max (or (get m key) 0) duration)))) {} (range len))
        max-duration (apply max (vals key-duration-map))
        max-duration? (fn [[key duration]]
                        (= duration max-duration))]
    (->> (filter max-duration? key-duration-map)
         (map first)
         (sort)
         (first))))
(map (partial apply slowest-key) ['([9,29,49,50] "cbcd") '([12,23,36,46,62] "spuda")])

;;1652
(defn decrypt [code k]
  (let [nums (into-array code)
        len (count nums)
        get-range (fn [start end]
                      (map #(code (rem % len)) (range (+ start len) (+ end len))))
        decode (fn [index]
                 (cond
                   (pos? k) (apply + (get-range (inc index) (+ index k 1)))
                   (neg? k) (apply + (get-range (+ index k) index))
                   :else 0))]
    (doseq [index (range len)]
      (aset nums index (decode index)))
    (vec nums)))
(map (partial apply decrypt) ['([5 7 1 4] 3) '([1 2 3 4] 0) '([2 4 9 3] -2)])

;;1640
(defn can-form-array [arr pieces]
  (let [piece-map (into {} (map (fn [%] [(first %) %]) pieces))]
    (= (flatten (map #(get piece-map %) arr)) arr)))
(map (partial apply can-form-array) ['([85] [[85]]) '([15 88] [[88] [15]]) '([49 18 16] [[16 18 49]]) '([91 4 64 78] [[78] [4 64] [91]]) '([1 3 5 7] [[2 4 6 8]])])

;;1700
(defn count-students [students sandwiches]
  (letfn [(take-lunch [students sandwiches]
            (cond
              (empty? students) 0
              (= (first students) (first sandwiches)) (take-lunch (subvec students 1) (subvec sandwiches 1))
              (= (count (set students)) 1) (count students)
              :else (take-lunch (conj (subvec students 1) (first students)) sandwiches)))]
   (take-lunch students sandwiches)))
(defn count-students [students sandwiches]
  (let [counter (make-array Long/TYPE 2)
        take-lunch (fn [num-students sandwich]
          (if (pos? (aget counter sandwich))
            (do
              (aset counter sandwich (dec (aget counter sandwich)))
              (dec num-students))
            (reduced num-students)))]
    (doseq [student students]
      (aset counter student (inc (aget counter student))))
    (reduce take-lunch (count students) sandwiches)))
(map (partial apply count-students) ['([1 1 0 0] [0 1 0 1]) '([1 1 1 0 0 1] [1 0 0 0 1 1])])

;;1694
(defn reformat-number [number]
  (let [digits (vec (re-seq #"\d" number))
        len (count digits)
        remainder (let [r (rem len 3)]
                    (if (= r 1)
                      4
                      r))
        reformat-3-digits (fn [digits]
                            (let [len (count digits)]
                              (reduce (fn [result index]
                                        (if (and (zero? (rem index 3))
                                                 (> index 0))
                                          (str result "-" (digits index))
                                          (str result (digits index)))) "" (range len))))
        reformat-4-digits (fn [digits]
                            (str (str/join "" (subvec digits 0 2)) "-" (str/join "" (subvec digits 2 4))))]
    (cond
      (= remainder 2) (if (= len 2)
                        (str/join "" digits)
                        (str (reformat-3-digits (subvec digits 0 (- len 2))) "-" (str/join "" digits)))
      (= remainder 4) (if (= len 4)
                        (reformat-4-digits digits)
                        (str (reformat-3-digits (subvec digits 0 (- len 4)))
                             "-"
                             (reformat-4-digits (subvec digits (- len 4) len))))
      :else (if (= len 3)
              (str/join "" digits)
              (reformat-3-digits digits)))))

(defn reformat-number [number]
  (letfn [(reformat-digits [digits]
            (let [len (count digits)]
              (cond
                (<= len 3) digits
                (= len 4) (str (subs digits 0 2) "-" (subs digits 2 4))
                :else (str (subs digits 0 3) "-" (reformat-digits (subs digits 3))))))]
    (->> (re-seq #"\d" number)
         (str/join "")
         (reformat-digits))))
(map reformat-number ["1-23-45 6" "123 4-567" "123 4-5678" "12" "--17-5 229 35-39475 "])

;;1710
(defn maximum-units [box-types truck-size]
  (let [box-types (vec (sort #(compare (last %2) (last %1)) box-types))]
    (letfn [(count-box [num-boxes total-units box-types]
              (let [[boxes units] (first box-types)]
                (cond
                  (empty? box-types) total-units
                  (= (inc num-boxes) truck-size) (+ total-units units)
                  (= boxes 1) (count-box (inc num-boxes) (+ total-units units) (subvec box-types 1))
                  :else (count-box (inc num-boxes) (+ total-units units) (vec (cons [(dec boxes) units] (subvec box-types 1)))))))]
      (count-box 0 0 box-types))))
(map (partial apply maximum-units) ['([[1 3] [2 2] [3 1]] 4) '([[5 10] [2 5] [4 7] [3 9]] 10)])

;;1736
(defn maximum-time [time]
  (let [digits (vec time)
        hour1 (cond
                (and (= (digits 0) \?) (= (digits 1) \?)) \2
                (and (= (digits 0) \?) (< (int (digits 1)) (int \4))) \2
                (and (= (digits 0) \?) (>= (int (digits 1)) (int \4))) \1
                :else (digits 0))
        hour2 (cond
                (= (digits 0) \0) \9
                (= (digits 0) \1) \9
                (= (digits 0) \2) \3
                :else \3)
        minute1 (if (= (digits 3) \?) \5 (digits 3))
        minute2 (if (= (digits 4) \?) \9 (digits 4))]
    (str hour1 hour2 ":" minute1 minute2)))
(map maximum-time ["2?:?0" "0?:3?" "1?:22"])

;;1752
(defn check [nums]
  (let [check' (fn [nums]
                 (let [max-value (apply max nums)
                       len (count nums)
                       new-nums (vec (concat nums nums))
                       index (.lastIndexOf new-nums max-value)
                       nums' (subvec new-nums (- index (dec len)) (inc index))]
                   (= (sort nums) nums')))]
    (cond
      (<= (count nums) 2) true
      (= (count (set nums)) 1) true
      :else (check' nums))))
(defn check [nums]
  (let [len (count nums)
        check' (fn [cnt index]
                 (if (> (nums index) (nums (rem (inc index) len)))
                   (inc cnt)
                   cnt))
       greater (reduce check' 0 (range len))]
    (<= greater 1)))
(map check [[3 4 5 1 2] [2 1 3 4] [1 2 3] [1 1 1] [2 1]])

;;1758
(defn min-operations [s]
  (let [len (count s)
        bits (vec s)
        count-operation (fn [[result0 result1] index]
                          (let [bit (bits index)]
                            (if (or (and (even? index) (= bit \0))
                                    (and (odd? index) (= bit \1)))
                              [result0 (inc result1)]
                              [(inc result0) result1])))
        [result0 result1] (reduce count-operation [0 0] (range len))]
    (min result0 result1)))
(map min-operations ["0100" "10" "1111"])

;;1779
(defn nearest-valid-point [x y points]
  (let [abs (fn [n] (if (neg? n) (- n) n))
        manhattan-distance (fn [[x' y']]
                             (+ (abs (- x x')) (abs (- y y'))))]
  (->> (filter (fn [[index [x' y']]] (or (= x x') (= y y'))) (map vector (range (count points)) points))
       (#(mapv (fn [[index [x' y']]] [index (manhattan-distance [x' y'])]) %))
       (#(filter (fn [[index distance]] (= distance (apply min (map last %)))) %))
       (#(or (first (first %)) -1)))))
(map (partial apply nearest-valid-point) ['(3 4 [[1 2] [3 1] [2 4] [2 3] [4 4]]) '(3 4 [[3 4]]) '(3 4 [[2 3]])])

;;1790
(defn are-almost-equal [s1 s2]
  (let [cs1 (vec s1)
        cs2 (vec s2)
    indexes (filter #(not= (cs1 %) (cs2 %)) (range (count s1)))]
    (cond
      (empty? indexes) true
      (and (= (count indexes) 2)
           (= (cs1 (first indexes)) (cs2 (last indexes)))
              (= (cs1 (last indexes)) (cs2 (first indexes)))) true
      :else false)))
(map (partial apply are-almost-equal) ['("bank" "kanb") '("attack" "defend") '("kelb" "kelb") '("abcd" "dcba")])

;;1805
(defn num-different-integers [word]
  (->> (re-seq #"\d+" word)
       (map #(Integer/parseInt %))
       (distinct)
       (count)))
(map num-different-integers ["a123bc34d8ef34" "leet1234code234" "a1b01c001"])

;;1863
(defn subset-xor-sum [nums]
  (let [n (count nums)]
  (* (apply bit-or nums) (bit-shift-left 1 (dec n)))))
(map subset-xor-sum [[1 3] [5 1 6] [3 4 5 6 7 8] [1 1 1 1 1 1]])

;;1893
(defn is-covered [ranges left right]
  (let [nums (make-array Long/TYPE 51)]
    (doseq [[start end] ranges]
      (doseq [i (range start (inc end))]
        (aset nums i (inc (aget nums i)))))
    (every? #(pos? (aget nums %)) (range left (inc right)))
  ))
(map (partial apply is-covered) ['([[1 2] [3 4] [5 6]] 2 5) '([[1 10] [10 20]] 21 21)])

;;1886
(defn find-rotation [mat target]
  (let [rotate (fn [mat]
                 (let [n (count mat)
                       matrix (make-array Long/TYPE n n)]
                   (doseq [r (range n) c (range n)]
                     (aset matrix c (- n 1 r) ((mat r) c)))
                   (mapv vec matrix)))
        check (fn [[result m] _]
                (let [m' (rotate m)]
                  (if (= target m')
                    (reduced [true m'])
                    [result m'])))]
    (first (reduce check [false mat] (range 4)))))
(map (partial apply find-rotation) ['([[0 1] [1 0]] [[1 0] [0 1]]) '([[0 1] [1 1]] [[1 0] [0 1]]) '([[0 0 0] [0 1 0] [1 1 1]] [[1 1 1] [0 1 0] [0 0 0]])])

;;1909
(defn can-be-increasing [nums]
  (let [len (count nums)
        increasing? (fn [nums] (apply < nums))
        ]
    (letfn [(can-be-increasing? [index deleted]
              (loop [index index deleted deleted])
              (if (>= index (dec len))
                true
                (let [x0 (nums (dec index))
                      x1 (nums index)
                      x2 (nums (inc index))]
                  (cond
                    (and (<= x0 x1) (> x1 x2) (> x2 x0)) (if deleted
                                                           false
                                                           (recur (+ index 2) true))
                    (< x0 x1 x2) (recur (inc index) deleted)
                    :else false))))]
      (cond
        (<= len 2) true
        (or (increasing? (rest nums)) (increasing? (pop nums))) true
        :else (can-be-increasing? 1 false)))))
(map can-be-increasing [[1 2 10 5 7] [2 3 1 2] [1 1 1] [1 2 3] [1 1] [962 23 27 555] [449 354 508 962] [100 21 100] [262 138 583] [1 2 5 10 7]])

;;1913
(defn max-product-difference [nums]
  (let [nums (vec (sort nums))
        len (count nums)
        max1 (nums (- len 1))
        max2 (nums (- len 2))
        min1 (first nums)
        min2 (second nums)]
    (- (* max1 max2) (* min1 min2))))
(defn max-product-difference [nums]
  (let [max-min (fn [[max1 max2 min1 min2] num]
                  (cond
                    (>= num max1) [num max1 min1 min2]
                    (and (< num max1) (or (nil? max2) (> num max2))) [max1 num min1 min2]
                    (<= num min1) [max1 max2 num min1]
                    (and (> num min1) (or (nil? min2) (< num min2))) [max1 max2 min1 num]
                    :else [max1 max2 min1 min2]))
        initial-values [(max (first nums) (second nums)) nil (min (first nums) (second nums)) nil]
        [max1 max2 min1 min2] (reduce max-min initial-values nums)]
    (- (* max1 max2) (* min1 min2))))
(map max-product-difference [[5 6 2 7 4] [4 2 5 9 7 4 8]])

;;2006
(defn count-k-difference [nums k]
  (let [nums (vec (sort nums))
        count-difference (fn [[result index-map] index]
                           (let [num (nums index)
                                 indexes (or (get index-map num) [])
                                 result' (if (empty? indexes)
                                           result
                                           (+ result (count indexes)))]
                             [result' (assoc index-map (+ num k) (conj (get index-map (+ num k)) index))]))]
    (first (reduce count-difference [0 {}] (range (count nums))))))
(map (partial apply count-k-difference) ['([1 2 2 1] 1) '([1 3] 3) '([3 2 1 5 4] 2)])

;;1971
(defn valid-path [n edges start end]
  (let [append (fn [m key value]
                 (assoc m key (conj (or (get m key) []) value)))
        vertice-map (reduce (fn [m [v1 v2]]
                              (append (append m v1 v2) v2 v1)) {} edges)]
    (letfn [(valid? [path start end]
              (reduce (fn [result vertice]
                        (cond
                          (= vertice end) (reduced true)
                          (contains? path vertice) (or result false)
                          :else (valid? (conj path vertice) vertice end))) false (get vertice-map start))
              )]
      (valid? #{start} start end))))
(map (partial apply valid-path) ['(3 [[0 1] [1 2] [2 0]] 0 2) '(6 [[0 1] [0 2] [3 5] [5 4] [4 3]] 0 5)])
;;1995
(defn count-quadruplets [nums]
  (let [special-quadruplet? (fn [[a b c d]]
                              (= (+ (nums a) (nums b) (nums c)) (nums d)))
        len (count nums)
        indexes (for [a (range (- len 3)) b (range (inc a) (- len 2)) c (range (inc b) (- len 1)) d (range (inc c) len)]
                  [a b c d])]
    (count (filter special-quadruplet? indexes))))
(map count-quadruplets [[1 2 3 6] [3 3 6 4 5] [1 1 1 3 5]])

;;2027
(defn minimum-moves [s]
  (let [cs (vec s)
        len (count s)
        get-min-moves (fn [[moves length x-counter] index]
                        (let [c (cs index)]
                          (cond
                            (= index (dec len)) [(if (or (pos? x-counter) (= c \X)) (inc moves) moves) 0 0]
                            (and (= length 0) (= c \X)) [moves 1 1]
                            (and (= length 0) (= c \O)) [moves 0 0]
                            (= length 2) [(inc moves) 0 0]
                            :else [moves (inc length) (if (= c \X) (inc x-counter) x-counter)])))]
    (first (reduce get-min-moves [0 0 0] (range len)))))

(defn minimum-moves [s]
  (letfn [(get-min-moves [s index]
            (loop [index index moves 0]
              (cond
                (>= index (count s)) moves
                (= (subs s index (inc index)) "O") (recur (inc index) moves)
                :else (recur (+ index 3) (inc moves)))))]
    (get-min-moves s 0)))
(map minimum-moves ["XXX" "XXOX" "OOOO"])

;;2032
(defn two-out-of-three [nums1 nums2 nums3]
  (let [intersection (fn [xs1 xs2] (set/intersection (set xs1) (set xs2)))
        nums [(intersection nums1 nums2) (intersection nums2 nums3) (intersection nums3 nums1)]]
    (vec (apply set/union nums))))
(map (partial apply two-out-of-three) ['([1 1 3 2] [2 3] [3]) '([3 1] [2 3] [1 2]) '([1 2 2] [4 3 3] [5])])

;;2047
(defn count-valid-words [sentence]
  (let [rule1? (fn [token]
                 (->> (vec token)
                      (filter #(not (or (Character/isLowerCase %) (contains? #{\- \. \! \,} %))))
                      (count)
                      (zero?)))
        rule2? (fn [token]
                 (let [len (count token)
                       cs (vec token)
                       first-index (.indexOf token "-")
                       last-index (.lastIndexOf token "-")
                       left-index (dec first-index)
                       right-index (inc last-index)]
                   (or (= first-index -1) (and (= first-index last-index) (>= left-index 0) (< right-index len)
                                               (Character/isLowerCase (cs left-index))
                                               (Character/isLowerCase (cs right-index))))
                                               ))
        rule3? (fn [token]
                 (let [cs (vec token)
                       num-marks (count (filter #(contains? #{\, \. \!} %) cs))]
                   (or (= num-marks 0) (and (= num-marks 1) (contains? #{\, \. \!} (last cs))))))
        valid? (fn [token]
                 (and (rule1? token) (rule2? token) (rule3? token)))]
    (->> (re-seq #"\S+" sentence)
         (filter valid?)
         (count))))
(map count-valid-words ["cat and  dog" "!this  1-s b8d!" "alice and  bob are playing stone-game10" "he bought 2 pencils, 3 erasers, and 1  pencil-sharpener."])
