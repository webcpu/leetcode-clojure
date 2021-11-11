(ns playground.core
  (:gen-class)
  (:require [clojure.string :as str]
            [clojure.set :as set]
            [clojure.math.combinatorics :as comb]
            ))
;  (:import (playground.core ExpenseCalculations)))
(require '[clojure.core.async :as async :refer (chan >!! <!!)])
(require '[clojure.test :as test :refer (deftest is)])
(require '[clojure.string :as str])

(import 'java.io.File)

(:use 'clojure.test)
;(use '[debux.core])

(declare two-sums)

(defn x1
  "I don't do a whole lot ... yet."
  [& args]
  ;(eval (read-string "(println (str 3))"))
  (def r (or (= 0 1) (= "yes" "no")))
  (def r (if (= 0 1) "no", "yes"))
  (def names ["Larry" "Doreen" "Hulk"])
  (def m {1 "Charlie" :lastname "Fisherman" 2 {3 "c"}}))
  ;(println (get-in m [2 3])))
  ;(println ({1 2} 1))
  ;(def nums [2 7 11 15])
  ;(def target 9)


(defn test-two-sums
  []
  (def nums [3 2 4])
  (def target 6)
  (println (two-sums nums target)))


(defn two-sums
  [nums target]
  (def dict
    (reduce
     (fn [m p]
       (assoc m (last p) (first p)))
     {}
     (map-indexed vector nums)))

  (defn found?
    [k]
    (and
     (get dict (- target k))
     (not= target (+ k k))))

  (defn search
    [[k1 index1]]
    (if (found? k1)
      (let
       [k2 (- target k1) index2 (get dict k2)]
        [index1 index2])
      []))

  (first
   (filter
    not-empty
    (map search dict))))

(defn length-of-longest-substring
  [s]

  (def t (seq ""))
  (def result "")
  (doseq [c (seq s)]
    (def t (conj t c))
    (if (and
         (= (count (set t)) (count t))
         (> (count t) (count result)))
      (def result (clojure.string/join t))))

  (count result))


(defn test-length-of-longest-substring
  []
  (let [s "abcabcbb"]
    (println (length-of-longest-substring s))))



(defn reverse-integer
  [x]
  (def sign
    (if (> x 0) 1 -1))
  (def y
    (Integer/parseInt
     (apply
      str
      (reverse (str (Math/abs x))))))
  (* sign y))

(defn test-reverse-integer
  []
  (def x -123)
  (println (reverse-integer x)))

(defn is-palindrome
  [x]
  (if (>= x 0)
    (= (str x) (clojure.string/reverse (str x)))
    nil))


(defn test-is-palindrome
  []
  (def x 121)
  (println (is-palindrome x)))


(defn roman-to-int
  [s]
  (def digits [])
  (def dict {\M 1000 \D 500 \C 100 \L 50 \X 10 \V 5 \I 1})
  (def cs (seq s))
  (defn to-int
    [ds]
    (def nums (map
               (fn [d] (get dict d))
               ds))
    (reduce + 0 nums))

  (defn convert-roman-to-int
    [r x]
    (if (> (get dict x) (get dict (last digits)))
      (+ r (to-int digits))
      (+ r (get dict x))))

  (defn process-roman
    [r x]
    (if (empty? digits)
      (do
        (def digits (conj digits x))
        r)
      (convert-roman-to-int r x)))

  (+ (reduce process-roman 0 cs)
     (to-int digits)))

(defn test-roman-to-int
  []
  (def s "LVIII")
  (println (roman-to-int s)))

;(defn is-valid [s]
;  (def dict {\) \( \] \[ \} \{})
;  (defn should-append [r c]
;    (or
;      (empty? r)
;      (= (get dict c) (last r))))
;  (defn process [r c]
;    (if (should-append r c)
;      (conj stack c)
;      (drop-last r)))
;  (empty? (reduce process [] (to-array s))))
;
;(defn test-is-valid []
;  (def s1 "()[]{}")
;  (def s2 "([)]")
;  (is-valid s2))

(defn remove-duplicates [nums]
  (def xs nums)
  (defn remove-duplicates' [j i]
    (if (> (xs i) (xs j))
      (do
        (def xs (assoc xs (inc j) (xs i)))
        (inc j))
      j))
  (def r (reduce remove-duplicates' 0 (range (count xs))))
  (println xs)
  (inc r))

(defn test-remove-duplicates []
  (def xs1 [1 1 2])
  (def xs2 [0 0 1 1 1 2 2 3 3 4])
  (println (remove-duplicates xs2)))

(defn test-doto []
  (doto (java.util.HashMap.)
    (.put "HOME" "/home/me")
    (.put "SRC" "src")))

(defn throw-catch [f]
  [(try
     (f)
     (catch ArithmeticException e "No dividing by zero!")
     (catch Exception e (str (.getMessage e)))
     (finally (println "returning... ")))])

(defn print-seq [s]
  (when (not (empty? s))
    (prn (first s))
    (recur (rest s))))
(defn new-name [[f m l]]
  (str l ", " f " " m))
(defn remove-element [nums v]

  (defn remove-element' [xs i j]
    (if (>= i j)
      xs
      (if (and (= (xs i) v) (not= (xs j) v))
        (remove-element' (assoc xs i (xs j) j (xs i)) (inc i) (dec j))
        (if (not= (xs i) v)
          (remove-element' xs (inc i) j)
          (remove-element' xs i (dec j))))))

;(remove-element' nums 0 (dec (count nums)))
  (concat (filter #(not= % v) nums) (filter #(= % v) nums)))

(defn f1 [[xs j] [i x]]
  (println xs)
  (println j)
  (println i)
  (println x))
(defn -main2
  [& args]
  ;(test-length-of-longest-substring)
  ;(test-reverse-integer)
  ;(test-is-palindrome)
  ;(test-is-valid)
  ;(test-remove-duplicates)
  ;(throw (Exception. "I done throwed"))
  ;(throw-catch #(/ 10 0))
  ;(print-seq [1 2 3])
  ;(test-doto))
  ;(f1 [[1 2 3] 45] [3 4])
  ;(new-name ["Guy" "Lewis" "Steele"])

  ;(println (remove-element [0 1 2 2 3 0 4 2] 2))
  ;(remove-element [3 2 2 3] 3)
  (def x 3.2M)
  (def y 3N)
  (class x)
  (class y)
  (unchecked-add Long/MAX_VALUE Long/MAX_VALUE)
  (def a (rationalize 1.0e50))
  (def b (rationalize -1.0e50))
  (def c (rationalize 17.0e00))
  (+ a (+ b c))
  (+ (+ a b) c)
  (identical? 'ab 'ab)
  (= 'ab 'ab)
  (str 'ab)
  (eval (read-string "(+ 1 2)"))
  (let [x (with-meta 'goat {:name true})]
    (:name (meta x)))

  (def nums (range 100 1000))

  ;(for [x (range 100 1) y (range 1000) z (range 1000)]
  (defn gcd [a b]
    (if (zero? b)
      a
      (recur b (mod a b))))

  ;(defn gcd3 [n]
  ;  (def result 1)
  ;  (def nums (range 1 n))
  ;  (max (for [x nums y nums z nums]
  ;         (if (< x y z)
  ;           (do (def result (max result (gcd z (gcd x y))))
  ;               result)
  ;           result
  ;           )))
  ;  )
  ;(gcd3 100)
  ;(clojure.math.combinatorics/permutations #(1 2 3))
  (defn permutations [s]
    (lazy-seq
     (if (seq (rest s))
       (apply concat
              (for [x s]
                (map #(cons x %) (permutations (remove #{x} s)))))
       [s])))

  (defn gcd [a b]
    (if (zero? b)
      a
      (recur b (mod a b))))

  (defn gcd3 [[a b c]]
    (gcd c (gcd a b)))

  (defn from-digits [digits]
    (->> digits
         (partition 3)
         (map (partial apply str))
         (map read-string)))

  (defn max-gcd []
    (->> (permutations (range 1 10))
         (map (comp gcd3 from-digits))
         (apply max)))

  ;(max-gcd)

  (defn strstr [haystack needle]
    (defn search [indice i]
      (def c (nth haystack i))

      (if (zero? (count needle))
        [0]
        (cond
          (and (< (count indice) (count needle)) (= (nth needle (count indice)) c)) (conj indice i)
          (= (count indice) (count needle)) indice
          true [])))

    (def results (reduce search [] (range (count haystack))))
    (cond
      (zero? (count needle)) 0
      (or (empty? results) (not= (count results) (count needle))) -1
      true (first results)))
  ;(println (strstr "hello" "ll"))
  ;(println (strstr "aaaaa" "bba"))
  ;(strstr "" "")

  ;(defn search [nums target start end]
  ;  (if (> start end)
  ;    []
  ;    (do
  ;      (def mid (int (Math/floor (/ (+ start end) 2))))
  ;      (def v (nth nums mid))
  ;      (cond
  ;        (= v target) [mid]
  ;        (<= start end) (concat (search nums target start (dec mid)) (search nums target (inc mid) en))))))
  ;(defn search-insert [nums target]
  ;  (search nums target 0 (dec (count nums))))

  (defn binary-search
    [coll ^long coll-size target]
    (let [cnt (dec coll-size)]
      (loop [low-idx 0 high-idx cnt]
        (if (> low-idx high-idx)
          (if (> (first coll) target) 0 (count coll))
          (let [mid-idx (quot (+ low-idx high-idx) 2) mid-val (coll mid-idx)]
            (cond
              (= mid-val target) mid-idx
              (< mid-val target) (recur (inc mid-idx) high-idx)
              (> mid-val target) (recur low-idx (dec mid-idx))))))))
  ;(search-insert [1 3 5 6] 5)
  ;(binary-search [1 3 5 6] 4 7)
  (vals {1 2 3 4})
  (def m (make-array Integer/TYPE 3 3 3))
  ;(pprint m)
  (def xs [1 2 3 8])
  (replace {1 \a 3 \b 8 \c}, xs)
  (def m {:a 1 :b 2})
  (empty? m)
  (filter #(= (first %) :b) m)
  (into (sorted-map) m)

  (defn length-of-last-word [s]
    (->> (clojure.string/split s #"\s+")
         (filter #(pos? (count %)))
         (last)))
  (def s "   fly me   to   the moon  ")
  ;(length-of-last-word s)
  (defn plus-one [coll]
    (defn add [xs x]
      (def s (+ x (first xs)))
      (cons (quot s 10) (cons (rem s 10) (rest xs))))
    (defn plus-one' [coll]
      (reduce add [1] (reverse coll)))
    (let [r (plus-one' coll)]
      (if (zero? (first r))
        (rest r)
        r)))
  ;(plus-one [9])
  (defn max-sub-array [nums]
    (defn sum [[max-value s] x]
      (cond
        (and (neg? x) (pos? (+ s x))) [max-value, (+ s x)]
        (and (neg? x) (not (pos? (+ s x)))) [(max max-value x) 0]
        :else [(max max-value (+ s x)), (+ s x)]))
    (first (reduce sum [(first nums) (first nums)] (rest nums))))
  ;(let [xss [ [1] [-2 1 -3 4 -1 2 1 -5 4] [5 4 -1 7 8] [-5 -6]]]
  ;  (map max-sub-array xss)
  ;  )
  (defn add-binary [a b]
    (defn to-digits [x]
      (map #(- (int %) (int \0)) (reverse x)))

    (defn add-bit [xs i]
      (if (< i (count xs))
        (nth xs i)
        0))

    (defn add-bits' [carry i]
      (let [xs (to-digits a) ys (to-digits b)]
        (+ carry (add-bit xs i) (add-bit ys i))))

    (defn add-bits [[result carry] i]
      (let [r (add-bits' carry i)]
        [(cons (rem r 2) result) (quot r 2)]))

    (defn add-carry [result]
      (if (pos? (last result))
        (cons (last result) (first result))
        (first result)))

    (let [indice (range (max (count a) (count b)))]
      (let [result (reduce add [[] 0] indice)]
        (add-carry result))))

  (let [inputs [["11", "1"] ["1010" "1011"]]]
    (map (partial apply add-binary) inputs))

  (defn my-sqrt [x]
    (defn my-sqrt' [x start end]
      (def m (quot (+ start end) 2))
      (let [s (* m m)]
        (cond
          (> start end) end
          (= s x) m
          (> s x) (my-sqrt' x start (dec m))
          (< s x) (my-sqrt' x (inc m) end))))
    (my-sqrt' x 0 x))
  ;(map my-sqrt [4 8 9 15 16 17])

  (defn climb-stairs [n]
    (cond
      (<= n 2) n
      true (+ (climb-stairs (dec n))
              (climb-stairs (- n 2)))))

;(map climb-stairs [2 3])
  (defn merge-arrays [nums1 m nums2 n]
    (defn dec-if-positive [x]
      (if (pos? x) (dec x) x))
    (declare merge')
    (defn merge-arrays' [i j tail]
      (let [a (aget nums1 i) b (aget nums2 j)]
        (when (pos? tail)
          (merge' a b i j tail))))
    (defn merge' [a b i j tail]
      (let [i' (if (> a b) (dec-if-positive i) i)
            j' (if (> a b) j (dec-if-positive j))
            value (if (> a b) a b)]
        (do
          (aset nums1 tail value)
          (merge-arrays' i' j' (dec tail)))))

    (merge-arrays' (dec m) (dec n) (dec (+ m n)))
    nums1)
  ;(pprint (merge-arrays (into-array [1 2 3 0 0 0]) 3 (into-array [2 5 6]) 3))
  (defn generate [n]
    (defn pascal [results n]
      (let [xs (last results)]
        (defn pascal' [result i]
          (let [x (+ (nth xs (dec i)) (nth xs i))]
            (vec (conj result x))))
        (let [r (reduce pascal' [] (range 1 (- n 1)))]
          (conj results (vec (cons 1 (conj r 1)))))))
    (if (= n 1)
      [[1]]
      (reduce pascal [[1]] (range 2 n))))
  ;(map generate [1 2 3 4 5])
  (defn get-row [n]
    (defn pascal [xs i]
      (defn pascal' [result j]
        (let [value (+ (nth xs j) (nth xs (dec j)))]
          (conj result value)))
      (let [r (reduce pascal' [] (range 1 i))]
        (vec (cons 1 (conj r 1)))))
    (if (= n 0)
      [1]
      (reduce pascal [1] (range 1 (inc n)))))
  (map get-row [0 1 2 3])
  (defn max-profit1 [prices]
    (def n (count prices))
    (def dp (make-array Integer/TYPE n n))
    (def profits (for [i (range (dec n)) j (range (inc i) n)]
                   (reduce #(max %1 (- (nth prices %2) (nth prices i))) 0 (range (inc i) n))))
    (apply max profits))

  (defn calculate-profit [[profit buy-price] price]
    (if (> price buy-price)
      [(max profit (- price buy-price)) buy-price]
      [profit price]))
  (defn max-profit [prices]
    (first
     (reduce calculate-profit [0 (first prices)] (rest prices))))
  ;(map max-profit [[7 1 5 3 6 4] [7 6 4 3 1]])

  (defn max-profit2 [prices]
    (first
     (reduce calculate-profit [0 (first prices)] (rest prices))))
  (map max-profit2 [[7 1 5 3 6 4] [1 2 3 4 5]])
  (defn is-palindrome2 [s]
    (defn remove-non-letters [s]
      (->> (clojure.string/lower-case s)
           (#(clojure.string/replace % #"[^a-z]" ""))))
    (let [s1 (remove-non-letters s)]
      (= (seq s1) (reverse s1))))
  ;(map is-palindrome2 ["A man, a plan, a canal: Panama", "race a car"])
  (defn single-number [xs]
    (let [m (frequencies xs)]
      (first (keys (filter #(= (last %) 1) m)))))
  ;(map single-number [[2 2 1] [4 1 2 1 2]])
  (defn two-sum-2 [xs target]
    (defn insert-index [m index x]
      (defn new-indice [m index x]
        (if (nil? (get m x))
          [(inc index)]
          (conj (get m x) (inc index))))
      (let [indice (new-indice m index x)]
        (assoc m x indice)))

    (def dict (reduce-kv insert-index {} xs))
    (defn valid? [indice target x]
      (or (empty? indice) (= target (+ x x))))
    (defn get-indice [target x]
      [(first (get dict x)) (first (get dict (- target x)))])
    (defn two-sum-2' [ks target]
      (def x (first ks))
      (let [indice (get dict (- target x))]
        (if (valid? indice target x)
          (two-sum-2' (rest ks) target)
          (get-indice target x))))
    (two-sum-2' (keys dict) target))
  ;(map (partial apply two-sum-2) [[[2 7 11 15] 9] [[2 3 4] 6] [[-1 0] -1]])

  (def letters (seq "ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
  (defn convert-to-title [n]
    (defn convert-to-title' [n]
      (if (<= n 26)
        [(nth letters (dec n))]
        (conj
         (convert-to-title' (quot n 26))
         (nth letters (dec (rem n 26))))))
    (clojure.string/join (convert-to-title' n)))

;(println letters)
  ;(map convert-to-title [1 28 701 2147483647])
  (defn trailing-zeros [n]
    (defn factor [n f result]
      (cond
        (not= 0 (rem n f)) result
        true (recur (quot n f) f (inc result))))

    (defn factors [n f]
      (let [factor' #(factor % f 0)
            xs (range 1 (inc n))]
        (reduce + (map factor' xs))))

    (let [fives (factors n 5)
          twos (factors n 2)]
      (min fives twos)))
  ;(map trailing-zeros [3 5 0 10])
  (defn majority-element [nums]
    (defn insert-value [m k v]
      (let [value (conj (or (get m v) []) k)]
        (assoc m v value)))
    (let [m (frequencies nums)]
      (let [inverted (reduce-kv insert-value {} m)]
        (first (get inverted (apply max (keys inverted)))))))
  ;(map majority-element [[3 2 3] [2 2 1 1 1 2 2]])
  (defn title-to-number [title]
    (defn add [r x]
      (let [digit (inc (- (int x) (int \A)))]
        (+ (* r 26) digit)))
    (reduce add 0 (seq title)))
  ;(map title-to-number ["A" "AB" "ZY" "FXSHRXW"])
  (defn reverse-bits [n]
    (defn integer-digits [n]
      (let [a (quot n 2) digit (rem n 2)]
        (if (zero? a)
          [digit]
          (conj (integer-digits a) digit))))

    (defn from-digits [xs]
      (let [accum #(+ (* %1 2) %2)]
        (reduce accum 0 xs)))

    (defn normalize [n]
      (let [max-value (bit-shift-left 1 32)
            complement-value (+ max-value n)]
        (if (neg? n)
          complement-value
          n)))

    (->> (normalize n)
         (integer-digits)
         (reverse)
         (from-digits)))
  ;(map reverse-bits [4 -3])
  (defn hamming-weight [num]
    (letfn [(hamming-weight' [num]
              (+ (rem num 2) (hamming-weight (quot num 2))))]
      (if (zero? num)
        0
        (hamming-weight' num))))
  ;(map hamming-weight [13 128])
  (defn happy? [n]
    (letfn
     [(integer-digits' [n]
        (if (zero? n)
          []
          (conj (integer-digits' (quot n 10)) (rem n 10))))

      (integer-digits [n]
        (if (zero? n)
          [0]
          (integer-digits' n)))

      (square-sum [digits]
        (reduce #(+ %1 (* %2 %2)) 0 digits))

      (calculate [n]
        (square-sum (integer-digits n)))

      (search [m n]
        (let [key (calculate n)]
          (cond
            (= 1 key) true
            (nil? (get m key)) (search (assoc m key true) key)
            true false)))]

      (search {n true} n)))
  ;(map happy? [19 2])
  (defn count-primes [n]
    (letfn [(prime?' [n i end]
              (cond
                (> i end) true
                (zero? (rem n i)) false
                true (prime?' n (inc i) end)))

            (prime? [n]
              (prime?' n 2 (int (Math/sqrt n))))

            (count-prime [sum n]
              (if (prime? n) (inc sum) sum))

            (count-primes' [n]
              (reduce count-prime 1 (range 3 (inc n))))]

      (cond
        (< n 2) 0
        (= n 2) 1
        true (count-primes' n))))
  ;(map count-primes [10 0 1])
  (defn isomorphic? [s t]
    (letfn
     [(create-map [s t]
        (let [xs (seq s) ys (seq t)]
          (reduce #(assoc %1 (nth xs %2) (nth ys %2)) {} (range (count xs)))))

      (translate [s t]
        (let [m (create-map s t)]
          (reduce #(conj %1 (get m %2)) [] (seq s))))

      (compare [s t]
        (= (translate s t) (seq t)))]

      (and (compare s t) (compare t s))))

;(map
  ;  (partial apply isomorphic?)
  ;  [["egg" "add"] ["foo" "bar"] ["paper" "title"]])
  (defn contains-duplicate [nums]
    (let [fs (frequencies nums)]
      (> (count nums) (count (keys fs)))))
  ;(map contains-duplicate [[1 2 3 1] [1 2 3 4]])
  (defn contains-nearby-duplicate [nums k]
    (letfn [(contains-nearby-duplicate' [nums i j k]
              (cond
                (> (- j i) k) false
                (= (nth nums i) (nth nums j)) true
                true (contains-nearby-duplicate' nums i (inc j) k)))
            (get-results [nums] (for [i (range 0 (- (count nums) k))]
                                  (contains-nearby-duplicate' nums i (inc i) k)))]
      (not (every? false? (get-results nums)))))

  ;(map (partial apply contains-nearby-duplicate) [[[1 2 3 1] 3] [[1 0 1 1] 1] [[1 2 3 1 2 3] 2]])
  (defn summary-ranges [nums]
    (letfn [(summary' [[results result] num]
              (cond
                (empty? result) [results [num]]
                (= (last result) (dec num)) [results (conj result num)]
                true [(conj results result) [num]]))

            (summary [nums]
              (let [[results result] (reduce summary' [[] []] nums)]
                (if (empty? result)
                  results
                  (conj results result))))

            (to-range [xs]
              (if (> (count xs) 1)
                (str (first xs) "->" (last xs))
                (str (first xs))))]
      (map to-range (summary nums))))
  ;(map summary-ranges [[0 1 2 4 5 7] [0 2 3 4 6 8 9] [] [-1] [0]])
  (defn is-power-of-two [n]
    (if (< n 1)
      false
      (zero? (bit-and n (dec n)))))
  ;(map is-power-of-two [1 16 3 4 5])
  (defn is-anagram [s t]
    (letfn [(to-kv-pairs [s]
              (sort (map #(str (first %) ":" (last %)) (seq (frequencies s)))))]
      (= (to-kv-pairs s) (to-kv-pairs t))
      ;(to-kv-pairs s))
      ))
  ;  (map (partial apply is-anagram) [["anagram" "nagaram"] ["rat" "car"]])
  (defn add-digits [num]
    (letfn [(integer-digits [n]
              (if (zero? (quot n 10))
                [(rem n 10)]
                (conj (integer-digits (quot n 10)) (rem n 10))))]
      (if (< num 10)
        num
        (add-digits (apply + (integer-digits num))))))
  ;(map add-digits [38 0])
  (defn missing-number [nums]
    (let [n (count nums)]
      (- (quot (* n (inc n)) 2) (apply + nums))))
  ;(map missing-number [[3 0 1] [0 1] [9 6 4 2 3 5 7 0 1] [0]])
  (defn move-zeroes [nums]
    (letfn [(fill [nums index i]
              (let [num (aget nums i)]
                (if (not= 0 num)
                  (do
                    (aset nums index num)
                    (inc index))
                  index)))
            (move [nums]
              (let [start (reduce (partial fill nums) 0 (range (count nums)))]
                (doseq [i (range start (count nums))]
                  (aset nums i 0))
                nums))]
      (if (< (count nums) 2)
        nums
        (vec (move (into-array nums))))))

;(map move-zeroes [[0 1 0 3 12] [0]])
  (defn word-pattern [pattern s]
    (letfn [(words [s]
              (clojure.string/split s #"\s"))
            (to-rules [cs ws]
              (reduce #(assoc %1 (nth cs %2) (nth ws %2)) {} (range (count cs))))
            (translate [s1 s2]
              (let [rules (to-rules s1 s2)]
                (map (partial get rules) s1)))]
      (let [s1 (vec pattern) s2 (words s)]
        (and (= (translate s1 s2) s2) (= (translate s2 s1) s1)))))
  ;(map (partial apply word-pattern) [["abba" "dog cat cat dog"] ["abba" "dog cat cat fish"] ["aaaa" "dog cat cat dog"] ["abba" "dog dog dog dog"]])
  (defn is-power-of-three [n]
    (letfn [(is-power-of-three' [n]
              (cond
                (= 1 n) true
                (not= 0 (rem n 3)) false
                true (recur (quot n 3))))]
      (if (< n 1)
        false
        (is-power-of-three' n))))

;  (map is-power-of-three [27 0 9 45])
  ;  (defn fib [n]
  ;    (Thread/sleep 2000)
  ;    (cond
  ;      (= n 1) 1
  ;      (= n 2) 1
  ;      true (last (reduce (fn [r _] [(nth r 1) (+ (nth r 0) (nth r 1))]) [1 1] (range 2 n)))))
  ;(let [begin-promise (promise)
  ;      end-promise (promise)]
  ;  (future
  ;    (deliver begin-promise (System/currentTimeMillis))
  ;    (Thread/sleep 1)
  ;    ;(map fib (range 1 200))
  ;    (deliver end-promise (System/currentTimeMillis)))
  ;  ;   (let [t (map fib (range 1 4))]
  ;  ;(println @begin-promise)
  ;  ;(println @end-promise)
  ;  (println (- @end-promise @begin-promise))
  ;  )
  ;(let [c (chan 1)]
  ;  (>!! c 5)
  ;  (println (<!! c))
  ;  )
  ;(some #{5 7} [1 8 7 5 3])
  ;(def pageview-stat (atom 0))
  ;(add-watch pageview-stat :pageview (fn [key agent old new] (println new)))
  ;(println pageview-stat)
  ;(swap! pageview-stat inc)
  ;(println pageview-stat)
  ;(Thread/sleep 500)
  ;(defn count-bits [n]
  ;  (letfn [(count-bits' [n]
  ;            (cond
  ;              (zero? n) 0
  ;              true (+ (bit-and n 1) (count-bits' (bit-shift-right n 1)))
  ;              )
  ;            )]
  ;    (map count-bits' (range 0 (inc n)))
  ;    )
  ;  )
  (defn count-bits [n]
    (letfn [(count-bits' [results i]
              (let [result (+ (nth results (quot i 2))
                              (bit-and i 1))]
                (conj results result)))]
      (reduce count-bits' [0] (range 1 (inc n)))))
  ;(map count-bits [2 5])
  (defn is-power-of-four [n]
    (let [powers (map #(bit-shift-left 1 (* 2 %)) (range 16))]
      (not (nil? (some #{n} (set powers))))))
  ;(map is-power-of-four [16 5 1])
  (defn reverse-str [s]
    (reduce #(cons %2 %) () (seq s)))
  ;(map reverse-string ["hello" "Hannah"])
  (defn reverse-vowels [s]
    (let [cs (into-array (seq s))
          indice (->> (seq s)
                      (map-indexed vector)
                      (filter #(some #{(first (seq (clojure.string/lower-case (last %))))} "aeiou")))]
      (letfn [(swap-values [cs indice _ i]
                (let [j (- (count indice) i 1)]
                  (let [p1 (nth indice i)
                        p2 (nth indice j)]
                    (aset cs (first p1) (last p2))
                    (aset cs (first p2) (last p1)))))]
        (let [len (quot (count indice) 2)]
          (reduce (partial swap-values cs indice) \a (range len))
          (clojure.string/join "" (vec cs))))))
  ;(map reverse-vowels ["hello" "leetcode"])
  ;(pprint (-> "a,b,c"
  ;      .toUpperCase
  ;      (.replace "A" "Z")
  ;      (.split ",")))
  (defn intersection [nums1 nums2]
    (vec (clojure.set/intersection (set nums1) (set nums2))))
  ;(map (partial apply intersection) [[[1 2 2 1] [2 2]] [[4 9 5] [9 4 9 8 4]]] )

  (defn intersect [nums1 nums2]
    (letfn [(update-freqs [[freqs xs] x]
              (let [freq (get freqs x)]
                (if (or (nil? freq) (zero? freq))
                  [freqs xs]
                  [(assoc freqs x (dec freq)) (conj xs x)])))]
      (last (reduce update-freqs [(frequencies nums2) []] nums1))))
  ;(map (partial apply intersect) [[[1 2 2 1] [2 2]] [[4 9 5] [9 4 9 8 4]]])
  (defn is-perfect-square [n]
    (letfn [(is-perfect-square' [n start end]
              (let [mid (quot (+ start end) 2)]
                (cond
                  (> start end) false
                  (= (* mid mid) n) true
                  (> (* mid mid) n) (is-perfect-square' n start (dec mid))
                  true (is-perfect-square' n (inc mid) end))))]

      (is-perfect-square' n 1 n)))
  ;(map is-perfect-square [16 14])
  (defn can-construct [ransom-note magzine]
    (let [freqs (frequencies magzine)]
      (letfn [(check-letter [freqs result [letter count2]]
                (let [count1 (get freqs letter)]
                  (if (or (nil? count1) (< count1 count2))
                    (reduced false)
                    result)))]
        (reduce (partial check-letter freqs)
                true
                (frequencies ransom-note)))))
  (deftest test-can-construct
    (is (= (map (partial apply can-construct) [["a" "b"] ["aa" "ab"] ["aa" "aab"]])
           [false false true]) "failed"))
  ;(test-can-construct)
  ;(defmacro unless [test then]
  ;  `(if (not ~test)
  ;     then))
  (defmacro unless [pred & expr]
    `(if (not ~pred) (do ~@expr)))

  ;  (pprint (macroexpand `(unless true (print "ab"))))
  ;(pprint (macroexpand '(unless false (print "ab") (print "cd"))))
  ;  (unless false (print "ab"))
  (defmacro def-logged-fn [fn-name args & body]
    `(defn ~fn-name ~args
       (let [now# (System/currentTimeMillis)]
         (println (str "[" now# "]"))
         ~@body)))

;(macroexpand '(def-logged-fn print-name [name] (reverse (seq name))))
  ;(def-logged-fn print-name [name] (reverse (seq name)))
  ;(print-name "abc")
  (defn debug [x]
    (println (str "x = " x))
    x)
  (defn first-uniq-char [s]
    (letfn [(add-index [result [i c]]
              (->> (or (get result c) [])
                   (#(conj % i))
                   (#(assoc result c %))))

            (count-indice [s]
              (->> (map-indexed vector s)
                   (reduce add-index {})))

            (first-uniq-char' [s]
              (let [m (count-indice s)
                    key (->> (keys m)
                             (filter #(= 1 (count (m %))))
                             (sort-by #(first (m %)))
                             (first))]
                (or (first (m key)) -1)))]
      (first-uniq-char' s)))
  ;(map first-uniq-char ["leetcode" "loveleetcode" "aabb"])
  (defn find-the-difference [s t]
    (let [freqs1 (frequencies s)
          freqs2 (frequencies t)
          not-equal (fn [key] (->> [freqs1 freqs2] (map #(get % key)) (apply not=)))]
      (->> (keys freqs2) (filter not-equal) (first))))

  ;(map (partial apply find-the-difference)
  ;     [["abcd" "abcde"] ["" "y"] ["a" "aa"] ["ae" "aea"]])
  ;)
  (defprotocol ExpenseCalculations
    (total [e]))
  (extend-protocol ExpenseCalculations nil
                   (total [e] 1))

  (extend-protocol ExpenseCalculations java.lang.Long
                   (total [e] "Long"))
  ;(total nil)
  ;(total 1)
  ;(defrecord Expense [date amount])
  ;(map->Expense {:date (System/currentTimeMillis) :amount 1})
  (defn is-subsequence [s t]
    (let [cs1 (seq s)
          cs2 (seq t)
          dp (make-array Long/TYPE (inc (count s)) (inc (count t)))]
      (letfn [(get-score [cs1 cs2 i j]
                (let [c1 (nth cs1 i)
                      c2 (nth cs2 j)]
                  (if (= c1 c2)
                    (inc (aget dp i j))
                    (aget dp (inc i) j))))
              (is-subsequence' [s t]
                (doseq [i (range (count s)) j (range (count t))]
                  (aset dp (inc i) (inc j) (get-score cs1 cs2 i j)))
                (aget dp (count s) (count t)))]
        (cond
          (> (count s) (count t)) false
          (= s t) true
          true (is-subsequence' s t)))))
  ;(map (partial apply is-subsequence) [["abc" "ahbgdc"] ["axc" "ahbgdc"]])
  ;(let [d1 (make-array Long/TYPE (inc (count "abc")) (inc (count "ahbgdc")))]
  ;  (pprint d1)
  ;  (aset d1 1 1 10)
  ;  )
  ;(let [the-array (make-array Long/TYPE 2 3)]
  ;  (doseq [i (range 2) j (range 3)]
  ;      (aset the-array i j (* i j)))
  ;  (pprint the-array)
  ;  )
  (defn read-binary-watch [n]
    (letfn [(all-time []
              (for [h (range 12) m (range 60)] [h m]))

            (ones [i]
              (let [rest-bits #(ones (bit-shift-right i 1))]
                (if (zero? i)
                  0
                  (+ (bit-and i 1) (rest-bits)))))
            (valid-time? [h m]
              (= n (+ (ones h) (ones m))))

            (time-string [h m]
              (str h ":" m))
            (add-time [results [h m]]
              (if (valid-time? h m)
                (conj results (time-string h m))
                results))]
      (reduce add-time [] (all-time))))
  ;(map read-binary-watch [1 9])
  (defn to-hex [num]
    (letfn [(twos-complement [n]
              (if (neg? n)
                (+ n (bit-shift-left 1 32))
                n))
            (add-digit [result n]
              (let [hexes (seq "0123456789abcdef")
                    get-digit #(nth hexes (rem % 16))
                    new-result #(cons (get-digit n) result)]
                (if (zero? n)
                  result
                  (add-digit (new-result) (quot n 16)))))
            (to-hex' [n]
              (add-digit [] n))
            (join [cs] (clojure.string/join "" cs))]
      (let [n (twos-complement num)]
        (if (zero? n)
          "0"
          (join (to-hex' n))))))
  ;(map to-hex [26 -1])
  (defn longest-palindrome [s]
    (let [letters (frequencies s)
          freqs (vals letters)
          evens (filter even? freqs)
          odds (sort > (filter odd? freqs))
          sum-of-odds (if (empty? odds)
                        0
                        (+ (first odds)
                           (apply + (map dec (rest odds)))))]
      (+ (apply + evens) sum-of-odds)))
  ;(map longest-palindrome ["abccccdd" "a" "bb"])
  (defn fizz-buzz [n]
    (letfn [(divisible? [a b]
              (zero? (rem a b)))
            (fizz-buzz' [n]
              (cond
                (divisible? n 15) "FizzBuzz"
                (divisible? n 3) "Fizz"
                (divisible? n 5) "Buzz"
                true (str n)))]
      (map fizz-buzz' (range 1 (inc n)))))
    ;(map fizz-buzz [3 5 15])
  (defn third-max [nums]
    (letfn [(maxs [[result xs] _]
              [(conj result (apply max xs))
               (filter #(not= % (apply max xs)) xs)])]
      (let [xs (distinct nums)
            rs (range (min (count xs) 3))
            results (first (reduce maxs [[] xs] rs))]
        (if (= 3 (count results))
          (last results)
          (first results)))))
  ;(map third-max [[3 2 1] [1 2] [2 2 3 1]])
  (defn add-strings [num1 num2]
    (let [ns1 (reverse (seq num1))
          ns2 (reverse (seq num2))
          len (max (count ns1) (count ns2))
          char-to-digit (fn [num ns i]
                          (if (< i (count num))
                            (- (int (nth ns i)) (int \0)) 0))
          add (fn [[digits carry] i]
                (let [s  (+ carry
                            (char-to-digit num1 ns1 i)
                            (char-to-digit num2 ns2 i))]
                  [(cons (rem s 10) digits) (quot s 10)]))
          results  (reduce add [[] 0] (range len))
          digits (if (zero? (last results))
                   (first results)
                   (cons (last results) (first results)))]
      (clojure.string/join "" digits)))
  ;(map (partial apply add-strings) [["11" "123"] ["456" "77"] ["0" "0"]])
  (defn count-segments [s]
    (->> (clojure.string/split s #"\s")
         (filter not-empty)
         (count)))
  ;(map count-segments ["Hello, my name is John" "Hello" "love live! mu'sic forever" ""])
  (defn arrange-coins [n]
    (letfn [(sum [i]
              (/ (* i (inc i)) 2))
            (arrange [n i]
              (if (< (sum i) n)
                i
                (recur n (dec i))))]
      (arrange n (int (Math/ceil (Math/sqrt n))))))
  ;(map arrange-coins [5 8 (bit-shift-left 1 31)])
  (defn find-disappeared-numbers [nums]
    (let [s1 (set nums)
          s2 (set (range 1 (inc (count nums))))]
      (->> (clojure.set/difference s2 s1)
           (vec)
           (sort))))
  ;(map find-disappeared-numbers [[4 3 2 7 8 2 3 1] [1 1]])
  (defn min-moves [nums]
    (let [a (reduce + nums)
          len (count nums)
          min-value (apply min nums)
          b (* len min-value)]
      (- a b)))
  ;(map min-moves [[1 2 3] [1 1 1]])
  (defn find-content-children [g s]
    (let [children g
          cookies s
          dp (make-array Long/TYPE 2 (inc (count g)))]
      (letfn [(update-score [i j]
                (let [delta (if (<= (nth g j) (nth s i)) 1 0)
                      score (max (aget dp 1 j) (+ (aget dp 0 j) delta))]
                  (aset dp 1 (inc j) score)))
              (count-score [i]
                (doseq  [j (range 0 (count g))]
                  (update-score i j))
                (doseq [j (range (inc (count g)))]
                  (aset dp 0 j (aget dp 1 j))
                  (aset dp 1 j 0)))]
        (doseq [i (range 0 (count s))]
          (count-score i))
        (aget dp 0 (count g)))))
  (map (partial apply find-content-children) [[[1 2 3] [1 1]] [[1 2] [1 2 3]]])
  (defn repeated-substring-pattern [s]
    (let [start 2
          len (count s)
          end (quot len 2)
          pattern? (fn [i]
                     (let [size (quot len i)
                           substring #(subs s (* % size) (* (inc %) size))
                           pattern (substring 0)
                           compare-pattern (fn [_ n]
                                             (if-not (= pattern (substring n))
                                               (reduced false)
                                               true))]
                       (if-not (zero? (rem len i))
                         false
                         (reduce compare-pattern true (range i)))))]
      (reduce #(if (pattern? %2) (reduced true) %1)
              false
              (range start (inc end)))))
  ;(map repeated-substring-pattern ["abab" "aba" "abcabcabcabc"])
  (defn hamming-distance [x y]
    (let [r (bit-xor x y)
          a (bit-shift-right x 1)
          b (bit-shift-right y 1)
          hamming-distance' #(+
                              (bit-and r 1)
                              (hamming-distance %1 %2))]
      (if (zero? r)
        0
        (hamming-distance' a b))))
  ;(map (partial apply hamming-distance) [[1 4] [3 1] [3 11]])
  (defn find-complement [num]
    (letfn [(find' [digits n]
              (cond
                (zero? n) (cons 1 digits)
                (= 1 n) digits
                true (find' (cons
                             (bit-xor 1 (bit-and 1 n))
                             digits)
                            (bit-shift-right n 1))))
            (find [n]
              (cond
                (zero? num) [1]
                (= 1 num) [0]
                true (vec (find' [] num))))]
      (reduce #(+ (bit-shift-left %1 1) %2) 0 (find num))))
                                        ;(map find-complement [5 1])
  (defn island-perimeter [grid]
    (let [rows (count grid)
          cols (count (grid 0))
          cordinates (for [y (range rows)
                           x (range cols)]
                       [y x])
          land-cordinates (filter #(= 1 (get-in grid %)) cordinates)]
      (letfn [(valid-cell? [y x]
                (and (>= x 0) (< x cols) (>= y 0) (< y rows)))
              (neighbors [y x]
                [[(inc y) x] [(dec y) x] [y (inc x)] [y (dec x)]])
              (neighbor-size [y x]
                (cond
                  (not (valid-cell? y x)) 1
                  (zero? (get-in grid [y x])) 1
                  true 0))
              (perimeter [y x]
                (reduce + 0 (map (partial apply neighbor-size) (neighbors y x))))]
        (reduce + 0 (map (partial apply perimeter) land-cordinates)))))
  ;(map island-perimeter [[[0 1 0 0] [1 1 1 0] [0 1 0 0] [1 1 0 0]] [[1]] [[1 0]]])
  (defn license-key-formatting [s k]
    (let [words (clojure.string/split s #"-")
          first-part (first words)
          join clojure.string/join
          cs (vec (seq (reduce #(str %1 %2) "" (rest words))))
          num-of-rest-parts (quot (count cs) k)
          subvector #(subvec cs (* % k) (* (inc %) k))
          rest-parts (map #(join "" (subvector %)) (range num-of-rest-parts))
          parts (cons first-part rest-parts)]
      (join "-" parts)
      ;(vec (range 1 10))
      ))
  ;(map (partial apply license-key-formatting) [["5F3Z-2e-9-w" 4] ["2-5g-3-J" 2]])

  ;; (defn find-palindomes []
  ;;   "find palindomes"
  ;;   (let [book "https://www.gutenberg.org/files/1513/1513-0.txt"
  ;;         palindome? #(= (reverse %) (seq %))]
  ;;     (->> (slurp book)
  ;;          (re-seq #"\w+")
  ;;          (map clojure.string/lower-case)
  ;;          (distinct)
  ;;          (sort)
  ;;          (filter palindome?)
  ;;          (sort-by count)
  ;;          (reverse)
  ;;          )))
  ;; ;(find-palindomes)
  (([[1 2 3] [2 3 4]] 0) 1)
  (nth '(1 2 3 4) 1)
  (hash-set 1 2 3 4 4)
  (set [1 2 3 4 4 4])
  ((or + -) 1 2 3)
  (defn flymaker
    ([x] 1)
    ([x y] 2))
  (flymaker 8)
;  (flymaker 8 9)
  (defn flymaker1 [{:keys [lat lon f]}]
    [lat lon f])
  (flymaker1 {:lat 1 :lon 2 :f 3})
  (loop [i 0]
    (if (>= i 5)
      8
      (recur (inc i))))
  (re-find #"^left-\w*" "left-eye a b")
  (clojure.string/replace "left-eye a b" #"^left-\w*" "right")
  (into [] '(4 5 6))
  (map (fn [_] (rand-int 3)) (range 10))
  (#(- % 9) 8)
  (set (map inc [1 1 2 2]))
  (map str (seq "abc") (seq "ABC") (seq "123"))
  (map #(% (range 4)) [count #(reduce + %)])

  (defn find-max-consecutive-ones [nums]
    (->> (map str nums)
         (reduce str "")
         (#(clojure.string/split % #"0"))
         (map count)
         (apply max)))

;(map find-max-consecutive-ones [[1 1 0 1 1 1] [1 0 1 1 0 1]])
  (defn construct-rectangle [area]
    (let [max-value (int (Math/sqrt area))
          divisible? #(if (zero? (rem area %2))
                        (reduced [(quot area %2) %2])
                        0)
          values (reverse (range 1 (inc max-value)))]
      (reduce divisible? 0 values)))

;(map construct-rectangle [4 37 122122])

  (defn find-poisoned-duration [time-series duration]
    (letfn [(reset-timer? [start t]
              (<= (- t start) (dec duration)))
            (count-poision-time' [result start t]
              (if (reset-timer? start t)
                [(+ result (inc (- start t))) t]
                [(+ result duration) t]))
            (count-poison-time [[result start] t]
              (if (nil? start)
                [result t]
                (count-poision-time' result start t)))
            (poison-time [] (first (reduce count-poison-time [0 nil] time-series)))]
      (+ duration (poison-time))))
  ;(map (partial apply find-poisoned-duration) [[[1 4] 2] [[1 2] 2]])

  (defn next-greater-element [nums1 nums2]
    (let [indices (map #(.indexOf nums2 %) nums1)]
      (letfn [(exist? [index]
                (or (neg? index)
                    (= (inc index) (count nums2))))
              (rest-indices [index]
                (range (inc index) (count nums2)))
              (find-index  [num index i]
                (if (> (get nums2 i) num)
                  (reduced (get nums2 i))
                  index))
              (next-greater-element' [index num]
                (if (exist? index)
                  -1
                  (reduce #(find-index num %1 %2) -1 (rest-indices index))))]
        (map next-greater-element' indices nums1))))
  ;(map (partial apply next-greater-element) [[[4 1 2] [1 3 4 2]] [[2 4] [1 2 3 4]]])
  ;; (let [identities [{:name "Batman"} {:name "Spiderman"}]]
  ;;   (map :name identities))
  ;; (take-while #(< % 8) (range 10))
  ;; (drop-while #(< % 6) (range 10))
  ;; (take-while #(<= % 8) (drop-while #(< % 6) (range 10)))
  ;; (some #(when (zero? (rem % 7)) %) (range 1 10))
  ;; ;(some #(when-not % [%]) [1 true false])
  ;; (take 8 (repeat "am"))
  ;; (take 3 (repeatedly #(rand-int 10)))
  ;; (apply conj [1] [2 3 4])
  ;; (conj [1] 2 3 4)
  ;; (condp #(= (clojure.string/upper-case %1 ) %2) "A"
  ;;   "a" "Best"
  ;;   "Not The Best")
  (defn find-words [words]
    (let [to-hashmap (fn [row]
                       (reduce #(assoc %1 %2 true) {} (seq row)))
          rows (map to-hashmap ["qwertyuiop" "asdfghjkl" "zxcvbnm"])
          check-letter? (fn [row result letter]
                          (if (nil? (get row letter))
                            (reduced false)
                            result))
          only-one-row (fn [w row]
                         (let [lower-case clojure.string/lower-case
                               word (seq (lower-case w))]
                           (reduce (partial check-letter? row) true word)))
          check-all-rows (fn [word]
                           (if (some true? (map #(only-one-row word %) rows))
                             true
                             false))]
      (filter check-all-rows words)))
  ;(map find-words [["Hello" "Alaska" "Dad" "Peace"] ["omk"] ["adsdf" "sfd"]])
  (defn convert-to-base7 [num]
    (letfn [(sign [] (if (neg? num) "-" ""))
            (abs [n] (if (neg? n) (* -1 n) n))
            (convert' [n]
              (if (zero? n)
                []
                (conj (convert' (quot n 7)) (rem n 7))))
            (convert [n]
              (if (zero? n)
                "0"
                (clojure.string/join "" (convert' n))))]
      (str (sign) (convert (abs num)))))
  ;(map convert-to-base7 [100 -7])
  (defn find-relative-ranks [score]
    (let [ranks #(into ["Gold Medal" "Silver Medal" "Bronze Medal"] (map str (range 4 (inc (count score)))))
          sorted-score (reverse (sort score))
          score-map (into {} (map-indexed #(hash-map %2 %1) sorted-score))
          get-rank #(get (ranks) (get score-map %))]
      (map get-rank score)))
  ;(map find-relative-ranks [[5 4 3 2 1] [10 3 8 9 4]])

  (defn check-perfect-number [num]
    (letfn [(add-factor [result x]
              (if (zero? (rem num x))
                (conj result  x (quot num x))
                result))
            (divisors []
              (range 2 (inc (int (Math/sqrt num)))))
            (factors [n]
              (reduce add-factor [1] (divisors)))]
      (= (reduce + (factors num)) num)))

  ;(map check-perfect-number [28 6 496 8128 2])
  (defn fib [n]
    (letfn [(fib' [n a b]
              (if (zero? n)
                b
                (fib' (dec n) b (+ a b))))]
      (case n
        0 0
        1 1
        (fib' (- n 1) 0 1))))
  ;(map fib [2 3 4 5])
  (defn detect-capital-user [word]
    (condp = word
      (clojure.string/upper-case word) true
      (clojure.string/lower-case word) true
      (clojure.string/join "" (clojure.string/capitalize (clojure.string/lower-case word))) true
      false))
  ;(map detect-capital-user ["USA" "FlaG"])
  ;; (defn find-LUS-length [a b]
  ;;   )
  ;; ;(map (partial apply find-LUS-length) [["aba" "cdc"] ["aaa" "bbb"] ["aaa" "aaa"]])
  (defn reverse-str [s k]
    (letfn [(reverse-str' [result i]
              (let [start (* i 2 k)
                    cs (vec s)
                    len (count s)]
                (cond
                  (>= (+ start k) len) (concat
                                        result
                                        (reverse (subvec cs start len)))
                  true (concat result
                               (reverse (subvec cs start (+ start k)))
                               (subvec cs (+ start k) (min len (+ start (* 2 k))))))))]

      (let [a (count s)
            b (* 2 k)
            n (if (zero? (rem a b))
                (quot a b)
                (inc (quot a b)))]
        (clojure.string/join "" (reduce reverse-str' [] (range n))))))
  ;(map (partial apply reverse-str) [["abcdefg" 2] ["abcd" 2]])
  (defn last-non-zero-digit [n]
    (letfn [(last-digit [i]
              (rem i 10))
            (extra-digits [digits]
              (let [ds (frequencies digits)
                    fives (or (get ds 5) 0)
                    twos  (or (get ds 2) 0)]
                (if (> fives twos)
                  (repeat (- fives twos) 5)
                  (repeat (- twos fives) 2))))
            (not-2-or-5? [i] (and (not= i 2) (not= i 5)))]
      (->> (map last-digit (range 1 (inc n)))
           (filter pos?)
           (#(concat (filter not-2-or-5? %) (extra-digits %)))
           (reduce #(last-digit (* %1 %2)) 1))))

;(map last-non-zero-digit [1 2 10 100 2020])
  ;(map (reduce #(rem (* %1 %2) 10) 1 [1 3 4 6 7 8 9])
  ;(get [8 4 2 6] (dec (rem 202 4)))
  (defn check-record [s]
    (let [als (vec (filter #(pos? (count %)) (str/split s #"P+")))
          split-by (fn [separator] (filter #(not (empty? %)) (reduce #(into %1 (str/split %2 separator)) [] als)))
          lates (split-by #"A+")]
      (and (< (or (get (frequencies s) \A) 0) 2)
           (empty? (filter #(>= (count %) 3) lates)))))

;(map check-record ["PPALLP" "PPALLL"])
  (defn reverse-words [s]
    (->> (str/split s #"\s")
         (map #(str/join "" (reverse %)))
         (#(str/join " " %))))

;(map reverse-words ["Let's take LeetCode contest" "God Ding"])
  (defn array-pair-sum [nums]
    (let [xs (sort nums)
          sum  (fn [result i]
                 (if (even? i)
                   (+ result (nth xs i))
                   result))]
      (reduce sum 0 (range (count xs)))))
  ;(map array-pair-sum [[1 4 3 2] [6 2 6 5 1 2]])
  ;; (defn matrix-reshape [mat r c]
  ;;   ((mat 0) 0)
  ;;   )
  (defn matrix-reshape [mat r c]
    (let [rows (count mat)
          cols (count (mat 0))
          m (make-array Long/TYPE r c)
          invalid? #(not= (* rows cols) (* %1 %2))
          reshape #(do
                     (doseq [i (range rows) j (range cols)]
                       (let [index (+ (* i cols) j)
                             row (quot index c)
                             col (rem index c)
                             value ((mat i) j)]
                         (aset m row col value)))
                     m)]

      (if (invalid? r c)
        mat
        (mapv vec (reshape)))))
  ;(map (partial apply matrix-reshape) [[[[1 2] [3 4]] 1 4] [[[1 2] [3 4]] 2 4]])
  (defn distribute-candies [candy-type]
    (let [types (count (set candy-type))
          max-allowed-types (quot (count candy-type) 2)]
      (min types max-allowed-types)))
  ;(map distribute-candies [[1 1 2 2 3 3] [1 1 2 3] [6 6 6 6]])
  (defn max-count [m n ops]
    (let [matrix (make-array Long/TYPE m n)]
      (letfn [(execute [r c]
                (doseq [i (range r) j (range c)]
                  (aset matrix i j (inc (aget matrix i j)))))
              (execute-ops [m n ops]
                (doseq [op ops]
                  (apply execute op)))
              (max-count' [m n ops]
                (let [freqs (do
                              (execute-ops m n ops)
                              (frequencies (into [] (flatten (mapv vec matrix)))))]
                  (last (last (into (sorted-map) freqs)))))]

        (if (empty? ops)
          (* m n)
          (max-count' m n ops)))))
                                        ;(map (partial apply max-count) [[3 3 [[2 2] [3 3]]]
                                        ;                                  [3 3 [[2 2] [3 3] [3 3] [3 3] [2 2] [3 3] [3 3] [3 3] [2 2] [3 3] [3 3] [3 3]]]
                                        ;                                  [3 3 []]])
  (defn find-restaurant [list1 list2]
    (let [common (clojure.set/intersection (set list1) (set list2))
          sum (fn [r] (+ (.indexOf list1 r) (.indexOf list2 r)))
          add-sum (fn [result r] (assoc result r (sum r)))
          interests (reduce add-sum {} common)
          least-sum (apply min (vals interests))
          least-index-sum? #(= least-sum (get interests %))]
      (vec (filter least-index-sum? (keys interests)))))
;;   (map (partial apply find-restaurant) [[["Shogun" "Tapioca Express" "Burger King" "KFC"] ["Piatti" "The Grill at Torrey Pines" "Hungry Hunter Steakhouse" "Shogun"]]
;;                                         [["Shogun" "Tapioca Express" "Burger King" "KFC"] ["KFC" "Burger King" "Tapioca Express" "Shogun"]]
;;                                         [["Shogun" "Tapioca Express" "Burger King" "KFC"] ["KNN" "KFC" "Burger King" "Tapioca Express" "Shogun"]]
;;                                         [["KFC"] ["KFC"]]
  ;;                                         ])
  ;; (defn t1 []
  (let [x "a"]
    (mapv str/upper-case [x]))
                                        ;(t1)
  (defn can-place-flowers [flowerbed n]
    (let [count-flower (fn [n] (quot (dec n) 2))
          plots  (map count (re-seq #"0+" (apply str flowerbed)))
          flowers (reduce + (mapv count-flower plots))]
      (>= flowers n)))
  ;(map (partial apply can-place-flowers) ['([1 0 0 0 1] 1) '([1 0 0 0 1] 2)])
  (defn maximum-product [nums]
    (let [xs (sort nums)]
      (max (apply * (conj (take 2 xs) (last xs))) (apply * (take 3 (sort > nums))))))
  ;(map maximum-product ['(1 2 3) '(1 2 3 4) '(-1 -2 -3)])
  (defn find-max-average [nums k]
    (let [initial (apply + (take k nums))
          slide-compare (fn [[result sum] i]
                          (let [new-sum (+ sum (get nums i) (* -1 (get nums (- i k))))]
                            (if (>= new-sum result)
                              [new-sum new-sum]
                              [result new-sum])))
          rs (range k (count nums))
          s (first (reduce slide-compare [initial initial] rs))]
      (float (/ s k))))
  ;(map (partial apply find-max-average) ['([1 12 -5 -6 50 3] 4) '([5] 1)])
  ;645
  (defn find-error-nums [nums]
    (let [s1 (set nums)
          s2 (set (range 1 (inc (count nums))))
          loss (first (clojure.set/difference s2 s1))
          duplicated (ffirst (filter #(> (last %) 1) (frequencies nums)))]
      [duplicated loss]))
  (map find-error-nums [[1 2 2 4] [1 1]])
  ;657
  (defn judge-circle [moves]
    (let [m (frequencies moves)]
      (and (= (m \U) (m \D)) (= (m \L) (m \R)))))
  ;(map judge-circle ["UD" "LL" "RRDD" "LDRRLRUULR"])
  ;661
  (defn image-smoother [img]
    (let [matrix (to-array-2d img)
          rows (count img)
          cols (count (img 0))

          valid? (fn [[i j]]
                   (and (>= i 0)
                        (>= j 0)
                        (< i rows)
                        (< j cols)))

          deltas (for [i (range -1 2) j (range -1 2)]
                   [i j])
          cells (fn [i j]
                  (let [plus (fn [[y x]] [(+ i y) (+ j x)])]
                    (mapv plus deltas)))

          avg (fn [xs] (quot (reduce + 0 xs) (count xs)))

          smooth' (fn [m i j]
                    (->> (cells i j)
                         (filter valid?)
                         (mapv (fn [[r c]] ((img r) c)))
                         (avg)))
          smooth (fn [m i j]
                   (let [value (smooth' m i j)]
                     (aset m i j value)))
          smooth-all #(doseq [i (range rows) j (range cols)]
                        (smooth matrix i j))]
      (smooth-all)
      (mapv vec matrix)))
  ;(map image-smoother [[[1 1 1] [1 0 1] [1 1 1]] [[100 200 100] [200 50 200] [100 200 100]]])
  674
  (defn find-length-of-LCIS [nums]
    (letfn [(increasing? [i]
              (and (< (nums (dec i)) (nums i)) (not= (inc i) (count nums))))
            (compare-length [result len]
              (if (> len result) [len 0] [result 0]))
            (find [[result len] i]
              (cond
                (zero? i) [1 1]
                (increasing? i) [result (inc len)]
                true (compare-length result len)))
            (find-length []
              (reduce find [0 0] (range (count nums))))]
      (first (find-length))))
  ;(mapv find-length-of-LCIS [[1 3 5 4 7] [2 2 2 2 2]])
  ;680
  (defn valid-palindrome [s]
    (letfn [(perfect-palindrome? [cs deleting i j]
              (if (= i j)
                true
                (valid-palindrome' deleting cs (inc i) (dec j))))
            (valid-palindrome' [cs deleting i j]
              (cond
                (= (inc i) j) true
                (= (get cs i) (get cs j)) (perfect-palindrome? cs deleting i j)
                (pos? deleting) (or (valid-palindrome' cs 1 (inc i) j)
                                    (valid-palindrome' cs 1 i (dec j)))
                :else false))]
      (valid-palindrome' (vec s) 0 0 (dec (count s)))))

;(map valid-palindrome ["aba" "abca" "abc"])
  ;682
  (defn cal-points [ops]
    (let [cal (fn [scores op]
                (case op
                  "+" (conj scores (apply + (take-last 2 scores)))
                  "D" (conj scores (* 2 (last scores)))
                  "C" (pop scores)
                  (conj scores (Integer/parseInt op))))]
      (apply + (reduce cal [] ops))))

;(map cal-points [["5" "2" "C" "D" "+"] ["5" "-2" "4" "C" "D" "9" "+" "+"] ["1"]])

  ;690
  (defrecord Employee [id importance subordinates])
  (defn get-importance [employees id]
    (letfn [(employee-table []
              (reduce #(assoc %1 (get %2 :id) %2) {} employees))
            (get-importance' [id]
              (let [employee (get (employee-table) id)
                    importance (get employee :importance)
                    subordinates (get employee :subordinates)
                    importances (+ importance (reduce + 0 (map get-importance' subordinates)))]
                (if (empty? subordinates)
                  importance
                  importances)))]

      (get-importance' id)
                                        ;      employee-table
      ))
  ;; (map get-importance
  ;;      (map #(map (partial apply ->Employee) %)
  ;;           [[[1 5 [2 3]] [2 3 []] [3 3 []]]
  ;;            [[1 2 [5]] [5 -3 []]]]) [1 5])
  ;693
  (defn has-alternating-bits [n]
    (letfn [(alternating-bits? [last-bit n]
              (cond
                (= n 1) (zero? last-bit)
                (zero? n) (= last-bit 1)
                (= 1 (bit-xor last-bit (bit-and n 1))) (alternating-bits? (bit-and n 1) (bit-shift-right n 1))
                :else false))]
      (alternating-bits? (bit-and n 1) (bit-shift-right n 1))))
  ;(mapv has-alternating-bits [5 7 11 10 3])
  ;696
  (defn count-binary-substrings1 [s]
    (let [cs (mapv #(Integer/parseInt %) (map str (seq s)))
          count-bin-string (fn [[result bits consecutive-len] i]
                             (let [bit (get cs i)]
                               (cond
                                 (empty? bits) [result [bit] 1]
                                 (and (= (count bits) (* 2 consecutive-len)) (pos? consecutive-len)) [(inc result) [bit] 1]
                                 (= (last bits) bit) [result (conj bits bit) (inc consecutive-len)]
                                 :else [result (conj bits bit) consecutive-len])))
          count-bin-strings #(first (reduce count-bin-string [0 [] 0] (range % (count s))))]
      (apply + (map count-bin-strings (range (count s))))))
  (defn count-binary-substrings [s]
    (let [ds (mapv #(Integer/parseInt %) (map str (seq s)))
          not-equal?' (fn [i]
                        (not= (get ds (dec i)) (get ds i)))
          not-equal? (fn [i]
                       (or (= i (count ds)) (not-equal?' i)))
          count-bs (fn [[result prev cur] i]
                     (cond
                       (zero? i) [result prev (inc cur)]
                       (not-equal? i) [(+ result (min prev cur)) cur 1]
                       :else [result prev (inc cur)]))]
      (first (reduce count-bs  [0 0 0] (range (inc (count ds)))))))
                                        ;(map count-binary-substrings ["00110011" "10101"])

  ;697
  (defn find-shortest-sub-array [nums]
    (let [degrees (frequencies nums)
          degree (apply max (vals degrees))
          max-degree-nums (map first (filter (fn [[_ value]] (= value degree)) degrees))
          get-shortest-sub-array-length (fn [num]
                                          (- (count nums)
                                             (.indexOf (reverse nums) num)
                                             (.indexOf nums num)))]
      (apply min (map get-shortest-sub-array-length max-degree-nums))))

;(map find-shortest-sub-array [[1,2,2,3,1] [1,2,2,3,1,4,2]])
  ;709
  (defn to-lower-case [s]
    (let [lowercase? #(Character/isLowerCase %)
          lowercase' #(+ (int %) (- (int \a) (int \A)))
          lowercase #(if (lowercase? %)
                       %
                       (lowercase' %))]
      (->> (seq s)
           (map lowercase)
           (map char)
           (str/join ""))))
  ;(map to-lower-case ["Hello" "here" "LOVELY"])
  ;717
  (defn is-one-bit-character [bits]
    (let [decode (fn [[result bits] bit]
                   (cond
                     (and (empty? bits) (zero? bit)) [(conj result [bit]) []]
                     (and (empty? bits) (= bit 1)) [result [1]]
                     (= bits [1]) [(conj result [1 bit]) []]
                     :else [result (conj bits bit)]))
          cs (first (reduce decode [[] []] bits))]
      (= (last cs) [0])))
  ;(map is-one-bit-character [[1 0 0] [1 1 1 0]])
                                        ;724
  (defn pivot-index [nums]
    (letfn [(pivot-index' [[index sum1 sum2] i]
              (let [num (get nums i)]
                (cond
                  (= sum1 (- sum2 num)) [i sum1 (- sum2 num)]
                  :else [index (+ sum1 num) (- sum2 num)])))]
      (let [sum (reduce + 0 nums)
            rs (range (count nums))]
        (first (reduce pivot-index' [-1 0 sum] rs)))))

  ;(map pivot-index [[1 7 3 6 5 6] [1 2 3] [2 1 -1]])
;728
  (defn self-dividing-numbers [left right]
    (letfn [(integer-digits [n]
              (let [a (quot n 10)
                    digit (rem n 10)]
                (cond
                  (zero? n) [0]
                  (zero? a) [digit]
                  :else (concat (integer-digits a) [digit]))))
            (self-dividing-digit? [n digit]
              (and (not= 0 digit)
                   (zero? (rem n digit))))
            (self-dividing-number? [n]
              (every? (partial self-dividing-digit? n) (integer-digits n)))
            (numbers []
              (range left (inc right)))]
      (filter self-dividing-number? (numbers))))
  ;(map (partial apply self-dividing-numbers) ['(1 22) '(47, 85)])
                                        ;733
  (defn flood-fill [image sr sc new-color]
    (let [matrix (to-array-2d image)
          rows (count image)
          cols (count (get image 0))
          valid-pixel? (fn [[r c]]
                         (and (>= r 0) (>= c 0) (< r rows) (< c cols)))
          get-color (fn [r c]
                      (aget matrix r c))
          same-color? (fn [[r c] old-color]
                        (= (get-color r c) old-color))
          plus (fn [[r c] [d1 d2]]
                 [(+ r d1)
                  (+ c d2)])
          connected-pixels (fn [[r c]]
                             (->> [[-1 0] [1 0] [0 -1] [0 1]]
                                  (map #(plus [r c] %))))]
      (letfn [(fill [[r c] old-color]
                (aset matrix r c new-color)
                (->> (connected-pixels [r c])
                     (filter valid-pixel?)
                     (filter #(same-color? % old-color))
                     ((fn [pixels] (doall (map #(fill % old-color) pixels))))))]
        (fill [sr sc] (get-color sr sc))
        (mapv vec matrix))))
  ;(map (partial apply flood-fill) ['([[1,1,1],[1,1,0],[1,0,1]] 1 1 2) '([[0,0,0],[0,0,0]] 0 0 2)])
                                        ;744
  (defn next-greatest-letter [letters target]
    (let [cs (sort (into [] (set letters)))
          find (fn [result c]
                 (if (> (int c) (int target))
                   (reduced [c])
                   result))]
      (first (reduce find [] cs))))
  ;(map (partial apply next-greatest-letter) ['([\c \f \j] \a) '([\c \f \j] \c) '([\c \f \j] \d) '([\c \f \j] \g) '([\c \f \j] \c)])
;746
  (defn min-cost-climbing-stairs [cost]
    (letfn [(climb [i]
              (cond
                (>= i (count cost)) 0
                true (+ (get cost i) (min (climb (inc i)) (climb (+ i 2))))))]
      (min (climb 0) (climb 1))))

  ;(map min-cost-climbing-stairs [[10,15,20] [1,100,1,1,1,100,1,1,100,1]])
  ;747
  (defn dominant-index [nums]
    (let [[a b] (take 2 (sort > nums))]
      (cond
        (= 1 (count nums)) 0
        (>= (quot a b) 2) (.indexOf nums a)
        :else -1)))
  ;(map dominant-index [[3 6 1 0] [1 2 3 4] [1]])
  ;748
  (defn shortest-completing-word [license-plate words]
    (let [license (->> license-plate
                       (re-seq #"[a-zA-Z]")
                       (#(str/join "" %))
                       (str/lower-case))]
      (letfn [(completing-word? [w]
                (let [dict1 (frequencies license)
                      dict2 (frequencies (str/lower-case w))]
                  (reduce (fn [result [letter value1]]
                            (let [value2 (get dict2 letter)]
                              (if (or (nil? value2) (< value2 value1))
                                (reduced false)
                                result))) true dict1)))]
        (->> words
             (filter completing-word?)
             ((fn [ws] (filter #(= (count %) (apply min (map count ws))) ws)))
             (first)))))

;;   (map (partial apply shortest-completing-word) ['("1s3 PSt" ["step","steps","stripe","stepple"])
;;                                                  '("1s3 456" ["looks","pest","stew","show"])
;;                                                  '("Ah71752" ["suggest","letter","of","husband","easy","education","drug","prevent","writer","old"])
;;                                                  '("OgEu755" ["enough","these","play","wide","wonder","box","arrive","money","tax","thus"])
;;                                                  '("iMSlpe4" ["claim","consumer","student","camera","public","never","wonder","simple","thought","use"])])
  ;762
  (defn count-prime-set-bits [left right]
    (letfn [(sqrt [n]
              (int (Math/sqrt n)))
            (prime? [n]
              (cond
                (or (= n 0) (= n 1)) false
                (= n 2) true
                :else (every? #(not= 0 (rem n %)) (range 2 (inc (sqrt n))))))
            (integer-digits' [n]
              (if (= n 0)
                []
                (concat (integer-digits' (bit-shift-right n 1)) [(bit-and n 1)])))
            (integer-digits [n]
              (if (zero? n)
                [0]
                (integer-digits' n)))]

      (->> (range left (inc right))
           (map integer-digits)
           (map #(apply + %))
           (filter prime?)
           (count))))
  ;(map (partial apply count-prime-set-bits) ['(6 10) '(10 15)])
  766
  (defn is-toeplitz-matrix [matrix]
    (let [rows (count matrix)
          cols (count (get matrix 0))
          valid-position? (fn [[r c]]
                            (and (>= r 0) (>= c 0) (< r rows) (< c cols)))
          compare-elements (fn [r c result i]
                             (let [r1 (+ r i) c1 (+ c i)]
                               (cond
                                 (not (valid-position? [r1 c1])) (reduced true)
                                 (= ((matrix r) c) ((matrix r1) c1)) result
                                 :else (reduced false))))
          same-elements? (fn [[r c]]
                           (let [rs (range 0 (max (- rows r) (- cols c)))]
                             (reduce #(compare-elements r c %1 %2) true rs)))
          first-row (map #(list 0 %) (range cols))
          first-col (map #(list % 0) (range rows))
          teoplitz-top-right? (every? same-elements? first-row)
          teoplitz-bottom-left? (every? same-elements? first-col)]
      (and teoplitz-top-right? teoplitz-bottom-left?)))

  ;(map is-toeplitz-matrix [[[1,2,3,4],[5,1,2,3],[9,5,1,2]] [[1,2],[2,2]]])
                                        ;771
  (defn num-jewels-in-stones [jewels stones]
    (let [jewels-map (frequencies jewels)
          stones-map (frequencies stones)
          sum (fn [s key] (if (nil? (get jewels-map key))
                            s
                            (+ s (get stones-map key))))]
      (reduce sum 0 (keys stones-map))))
  ;(map (partial apply num-jewels-in-stones) ['("aA" "aAAbbbb") '("z" "ZZ")])
  ;796
  (defn rotate-string1 [s goal]
    (let [xs (vec s)
          ys (vec goal)
          rotate (fn [i]
                   (concat (subvec xs i) (subvec xs 0 i)))
          compare-string (fn [result i]
                           (if (= ys (rotate i))
                             (reduced true)
                             false))]
      (reduce compare-string false (range 1 (count s)))))

  (defn rotate-string [s goal]
    (str/includes? goal (str s s)))
  ;(map (partial apply rotate-string) ['("abcde" "cdeab") '("abcde" "abced")])
  ;804
  (defn unique-morse-representations
    [words]
    (let [morse [".-" "-..." "-.-." "-.." "." "..-." "--." "...." ".." ".---" "-.-" ".-.." "--" "-." "---" ".--." "--.-" ".-." "..." "-" "..-" "...-" ".--" "-..-" "-.--" "--.."]
          insert (fn [m i] (assoc m (char (+ (int \a) i)) (get morse i)))
          morse-map (reduce insert {} (range 26))
          encode (fn [word] (map #(get morse-map %) (seq word)))]

      (->> words
           (map encode)
           (map #(str/join "" %))
           (set)
           (count))))

;(map unique-morse-representations [["gin" "zen" "gig" "msg"] ["a"]])
  ;806
  (defn number-of-lines [widths s]
    (let [width-map (reduce #(assoc %1 (char (+ (int \a) %2)) (get widths %2)) {} (range 26))
          get-width (fn [c]
                      (get width-map c))
          insert (fn [[lines pixels] c]
                   (let [width (get-width c)
                         new-pixels (+ pixels width)]
                     (cond
                       (> new-pixels 100) [(inc lines) width]
                       (= new-pixels 100) [(inc lines) 0]
                       :else [lines new-pixels])))
          [lines pixels] (reduce insert [0 0] (seq s))]
      [(+ lines (if (zero? pixels) 0 1)) pixels]))

;;   (map (partial apply number-of-lines)
;;        ['([10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10] "abcdefghijklmnopqrstuvwxyz")
;;         '([4 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10] "bbbcccdddaaa")])
                                        ;812
  (defn largest-triangle-area [points]
    (let [len (count points)
          triangles
          (for [i (range (- len 2))
                j (range (inc i) (- len 1))
                k (range (inc j) len)]
            [(points i) (points j) (points k)])

          sqrt #(Math/sqrt (double %))
          length (fn [x y] (sqrt (+ (* x x) (* y y))))
          dot (fn [[vx vy] [wx wy]] (+ (* vx wx) (* vy wy)))
          area (fn [[[ix iy] [jx jy] [kx ky]]]
                 (let [[vx vy] [(- jx ix) (- jy iy)]
                       [wx wy] [(- kx ix) (- ky iy)]
                       l1 (length vx vy)
                       l2 (length wx wy)
                       cos (/ (double (dot [vx vy] [wx wy])) (* l1 l2))]
                   (* l1 l2 0.5 (sqrt (- 1 (* cos cos))))))]

      (->> triangles
           (mapv area)
           (apply max))))
  ;(map largest-triangle-area [[[0 0] [0 1] [1 0] [0 2] [2 0]] [[1 0] [0 0] [0 1]]])
  (defn most-common-word [paragraph banned]
    (->> paragraph
         (re-seq #"\w+")
         (map str/lower-case)
         (filter #(empty? (clojure.set/intersection (set banned) (set [%]))))
         (frequencies)
         (sort-by last)
         (last)
         (first)))

  (map (partial apply most-common-word) ['("Bob hit a ball, the hit BALL flew far after it was hit." ["hit"])
                                         '("a." [])])
;821
  (defn shortest-to-char [s c]
    (let [cs (vec s)
          distances (long-array (count s) Long/MAX_VALUE)
          update (fn [i rs]
                   (reduce (fn [r offset]
                             (let [distance #(aget distances (+ i offset))
                                   new-index (+ i offset)
                                   absolute-offset (long (Math/abs offset))
                                   invalid-index? (or (< new-index 0) (>= new-index (count s)))
                                   update-distance  #(aset distances new-index absolute-offset)]
                               (cond
                                 invalid-index? (reduced true)
                                 (zero? (distance)) (reduced true)
                                 (< absolute-offset (distance)) (do (update-distance) true)
                                 :else (reduced true)))) true rs))
          update-both (fn [i]
                        (aset distances i 0)
                        (update i (next (range)))
                        (update i (reverse (range (* -1 i) 0))))]
      (->> (range (count s))
           (filter #(= (get cs %) c))
           (mapv update-both))
      (vec distances)))

  ;(map (partial apply shortest-to-char) ['("loveleetcode" \e) '("aaab" \b)])
  ;824
  (defn to-goat-latin [sentence]
    (let [transform1 (fn [word]
                       (let [cs (seq word)
                             first-letter (first cs)
                             letter (first (seq (str/lower-case word)))
                             index (.indexOf (seq "aeiou") letter)]
                         (if (>= index 0)
                           word
                           (str (str/join "" (rest cs)) first-letter))))

          transform (fn [index word]
                      (->> word
                           (transform1)
                           (#(str % "ma"))
                           (#(str % (str/join "" (take index (repeat \a)))))))
          words (re-seq #"\w+" sentence)]
      (mapv transform (range 1 (inc (count words))) words)))

  (map to-goat-latin ["I speak Goat Latin" "The quick brown fox jumped over the lazy dog"])
  ;830
  (defn large-group-positions1 [s]
    (let [cs (vec s)
          rs (range 1 (inc (count s)))
          find (fn [indices i]
                 (cond
                   (= i (count s)) (conj indices (dec i))
                   (= (get cs (dec i)) (get cs i)) indices
                   :else (conj indices (dec i))))
          end-indices (vec (reduce find [] rs))
          add-index (fn [result index] (concat result [index (inc index)]))
          indices (vec (reduce add-index [0] end-indices))
          full-indices (if (>= (last indices) (count s))
                         (pop indices)
                         (conj indices (dec (count s))))]

      (->> (partition 2 full-indices)
           (map vec)
           (filter (fn [[start end]] (>= (- end start) 2)))
           (vec))))

  ;; (defn large-group-positions [s]
  ;;   (let [cs (vec s)]
  ;;     (loop [i 0 j i]
  ;;       (if (< i (count s))
  ;;         (while ()
  ;;           (if (k)))))))

                                        ;(map large-group-positions ["abbxxxxzzy" "abc" "abcdddeeeeaabbbcd" "aba"])
                                        ;832
  (defn flip-and-invert-image [image]
    (->> image
         (map reverse)
         (map #(map (fn [bit] (bit-xor bit 1)) %))
         (mapv vec)))

;(map flip-and-invert-image [[[1 1 0] [1 0 1] [0 0 0]] [[1 1 0 0] [1 0 0 1] [0 1 1 1] [1 0 1 0]]])
  ;836
  (defn is-rectangle-overlap [rec1 rec2]
    (let [[left right] (if (< (rec1 0) (rec2 0))
                         [rec1 rec2]
                         [rec2 rec1])
          [up down] (if (< (rec1 1) (rec2 1))
                      [rec2 rec1]
                      [rec1 rec2])]
      (and (< (left 0) (right 0))
           (< (right 0) (left 2))
           (< (down 1) (up 1))
           (< (up 1) (down 3)))))

;(map (partial apply is-rectangle-overlap) ['([0 0 2 2] [1 1 3 3]) '([0 0 1 1] [1 0 2 1]) '([0 0 1 1] [2 2 3 3])])
  ;844
  (defn backspace-compare [s t]
    (let [interpret (fn [s]
                      (reduce #(cond
                                 (and (= %2 \#) (empty? %1)) []
                                 (= %2 \#) (pop %1)
                                 :else (conj %1 %2)) [] (vec s)))]
      (= (interpret s) (interpret t))))

;(map (partial apply backspace-compare) ['("ab#c" "ad#c") '("ab##" "c#d#") '("a##c" "#a#c") '("a#c" "b")])
;852
  (defn peak-index-in-mountain-array [arr]
    ;(.indexOf arr (apply max arr))
    (letfn [(peak [l r]
              (let [m (quot (+ l r) 2)]
                (cond
                  (>= l r) l
                  (< (get arr m) (get arr (inc m))) (peak (inc m) r)
                  (>= (get arr m) (get arr (inc m))) (peak l m))))]
      (peak 0 (dec (count arr)))))
  ;(map peak-index-in-mountain-array [[0 1 0] [0 2 1 0] [0 10 5 2] [3 4 5 1] [24 69 100 99 79 78 67 36 26 19]])
  ;859
  (defn buddy-strings [s goal]
    (let [cs1 (vec s) cs2 (vec goal)
          insert (fn [result i]
                   (let [c1 (cs1 i) c2 (cs2 i)]
                     (if (not= c1 c2)
                       (if (= 2 (count result))
                         (reduced (conj result [c1 c2]))
                         (conj result [c1 c2]))
                       result)))
          differences (reduce insert [] (range (count s)))
          equal-buddy-strings? (>= (apply max (vals (frequencies s))) 2)
          equal-length-buddy-strings? (if (= (count differences) 2)
                                        (= (first differences) (vec (reverse (last differences))))
                                        false)]

      (cond
        (= s goal) equal-buddy-strings?
        (= (count s) (count goal)) equal-length-buddy-strings?
        :else false)))
  ;(map (partial apply buddy-strings) ['("ab" "ba") '("ab" "ab") '("aa" "aa") '("aaaaaaabc" "aaaaaaacb")])
  ;860
  (defn lemonade-change [bills]
    (letfn [(take-money [my-bills bill]
              (update my-bills bill inc))
            (give-change' [[success my-bills] bill]
              (let [change (- bill 5)
                    fives (get my-bills 5)
                    tens (get my-bills 10)]
                (cond
                  (zero? change) [true my-bills]
                  (and (= change 5) (pos? fives)) [true (assoc my-bills 5 (dec fives))]
                  (and (= change 15) (pos? fives) (pos? tens)) [true (assoc (assoc my-bills 5 (dec fives)) 10 (dec tens))]
                  (and (= change 15) (>= fives 3)) [true (assoc my-bills 5 (- fives 3))]
                  :else [false my-bills])))
            (give-change [[success my-bills] bill]
              (give-change' [success (take-money my-bills bill)] bill))]
      (first (reduce give-change [true {5 0 10 0 20 0}] bills))))
  (map lemonade-change [[5 5 5 10 20] [5 5 10 10 20] [5 5 10] [10 10]])
                                        ;867
  (defn transpose [matrix]
    (let [rows (count matrix)
          cols (count (matrix 0))
          m (make-array Long/TYPE cols rows)]
      (doseq [i (range rows) j (range cols)]
        (aset m j i ((matrix i) j)))
      (mapv vec m)))

  (map transpose [[[1 2 3] [4 5 6] [7 8 9]] [[1 2 3] [4 5 6]]])
                                        ;868
  (defn binary-gap [n]
    (letfn [(integer-digits' [n]
              (if (= n 1)
                [1]
                (conj (integer-digits' (quot n 2)) (rem n 2))))
            (integer-digits [n]
              (if (zero? n)
                [0]
                (integer-digits' n)))
            (max-gap [indices]
              (let [gap (fn [result i]
                          (max result
                               (- (get indices i)
                                  (get indices (dec i)))))
                    rs (range 1 (count indices))]
                (reduce gap 0 rs)))]

      (let [digits (integer-digits n)]
        (->> digits
             (#(map vector (range (count %)) %))
             (filter #(= (last %) 1))
             (mapv first)
             (max-gap)))))

  (map binary-gap [22 5 6 8 1])
  (defn robot-sim [commands obstacles]
    (let [obstacle-map (reduce #(assoc %1 %2 true) {} obstacles)
          euclid-distance (fn [x y]
                            (+ (* x x) (* y y)))
          directions [[0 1] [1 0] [0 -1] [-1 0]]
          update-direction-index (fn [index cmd]
                                   (if (= cmd -2)
                                     (if (zero? index) 3 (dec index))
                                     (rem (inc index) 4)))
          obstacle? (fn [x y] (if (nil? (get obstacles [x y]))
                                false
                                true))
          move (fn [x y index cmd]
                 (let [[v w] (directions index)
                       [x' y'] [(+ x (* cmd v)) (+ y (* cmd w))]
                       move' (fn [result [x1 y1]]
                               (if (obstacle? x1 y1)
                                 (reduced [(+ x1 (* -1 v)) (+ y1 (* -1 w))])
                                 result))
                       cordinates (map (fn [i]
                                         [(+ x (* i v)) (+ y (* i w))])
                                       (range 1 (inc cmd)))]
                   (reduce move' [x' y'] cordinates)))
          execute (fn [[x y index max-distance] cmd]
                    (if (or (= cmd -1) (= cmd -2))
                      [x y (update-direction-index index cmd) max-distance]
                      (let [[x1 y1] (move x y index cmd)]
                        [x1 y1 index (max max-distance (euclid-distance x1 y1))])))]
      (last (reduce execute [0 0 0 0] commands))))
                                        ;(map (partial apply robot-sim) ['([4 -1 3] []) '([4 -1 4 -2 4] [[2 4]])])
                                        ;883
  (defn projection-area [grid]
    (let [sum #(apply + %)
          maximum #(apply max %)
          transpose (fn [m]
                      (let [rows (count m)
                            cols (count (m 0))
                            matrix (make-array Long/TYPE cols rows)]
                        (doseq [i (range cols) j (range rows)]
                          (aset matrix i j ((m j) i)))
                        (mapv vec matrix)))
          grid' (transpose grid)
          xy (count (filter pos? (flatten grid)))
          xz (sum (map maximum grid))
          yz (sum (map maximum grid'))]
      (+ xy xz yz)))

  ;(map projection-area [[[1 2] [3 4]] [[2]] [[1 0] [0 2]] [[1 1 1] [1 0 1] [1 1 1]] [[2 2 2] [2 1 2] [2 2 2]]])
  ;884
  (defn uncommon-from-sentences [s1 s2]
    (let [words #(re-seq #"\w+" %)
          unique-words (fn [s] (->> (words s)
                                    (frequencies)
                                    (filter #(= 1 (last %)))
                                    (map first)
                                    (set)))
          ws1 (unique-words s1)
          ws2 (unique-words s2)]
      (->> (map clojure.set/difference [ws1 ws2] [ws2 ws1])
           (mapv vec)
           (flatten))))

  (map (partial apply uncommon-from-sentences) ['("this apple is sweet" "this apple is sour") '("apple apple" "banana")])
                                        ;888
  (defn fair-candy-swap [alice-sizes bob-sizes]
    (let [total #(apply + %)
          alice-map (frequencies alice-sizes)
          bob-map (frequencies bob-sizes)
          alice-candies (total alice-sizes)
          bob-candies (total bob-sizes)
          delta (- alice-candies bob-candies)
          get-candy (fn [candy] (- candy (quot delta 2)))
          exchange (fn [result candy] (if (nil? (get bob-map (get-candy candy)))
                                        result
                                        (reduced [candy (get-candy candy)])))]
      (reduce exchange [] (keys alice-map))))
  (map (partial apply fair-candy-swap) ['([1 1] [2 2]) '([1 2] [2 3]) '([2] [1 3]) '([1 2 5] [2 4])])
  ;892
  (defn surface-area [grid]
    (let [grid->cubes (fn [m]
                        (let [rows (count m)
                              cols (count (m 0))
                              add-cubes (fn [i j]
                                          (mapv (fn [k] [i j k]) (range ((m i) j))))
                              nested-cubes (for [i (range rows) j (range cols)] (add-cubes i j))]
                          (vec (reduce concat [] nested-cubes))))
          cubes (grid->cubes grid)
          cube-set (set cubes)
          empty-cell? (fn [[x y z]]
                        (if (nil? (get cube-set [x y z]))
                          true
                          false))
          deltas [[-1 0 0] [1 0 0] [0 -1 0] [0 1 0] [0 0 -1] [0 0 1]]
          plus (fn [[x1 y1 z1] [x2 y2 z2]] (mapv + [x1 y1 z1] [x2 y2 z2]))
          neighbors (fn [[i j k]]
                      (mapv #(plus [i j k] %) deltas))
          cube-surface-area (fn [[i j k]]
                              (count (filter empty-cell? (neighbors [i j k]))))]
      (apply + (mapv cube-surface-area cubes))))
  (map surface-area [[[2]] [[1 2] [3 4]] [[1 0] [0 2]] [[1 1 1] [1 0 1] [1 1 1]] [[2 2 2] [2 1 2] [2 2 2]]])
  (defn is-monotonic [nums]
    ;; (if (<= (first nums) (last nums))
    ;;   (= (sort nums) nums)
    ;;   (= (sort > nums) nums))
    (let [check (fn [[increasing decreasing] i]
                  [(and increasing (<= (nums (dec i)) (nums i)))
                   (and decreasing (>= (nums (dec i)) (nums i)))])
          [increasing decreasing] (reduce check [true true] (range 1 (count nums)))]
      (or increasing decreasing)))
  (map is-monotonic [[1 2 2 3] [6 5 4 4] [1 3 2] [1 2 4 5] [1 1 1]])
  ;905
  (defn sort-array-by-partity [nums]
    (concat (filter even? nums) (filter odd? nums)))
  ;(map sort-array-by-partity [[3 1 2 4] [0]])
  ;908
  (defn smallest-range-i [nums k]
    (let [max-num (apply max nums)
          min-num (apply min nums)
          delta (- max-num min-num (* 2 k))]
      (max delta 0)))
  ;(map (partial apply smallest-range-i) ['([1] 0) '([0 10] 2) '([1 3 6] 3)])
                                        ;914
  (defn has-groups-size-x [deck]
    (let [cards-list (vals (frequencies deck))
          min-cards (apply min cards-list)]
      (and (> min-cards 1) (every? #(zero? (rem % min-cards)) cards-list))))
  ;(map has-groups-size-x [[1 2 3 4 4 3 2 1] [1 1 1 2 2 2 3 3] [1] [1 1] [1 1 2 2 2 2]])
  (long (.abs (biginteger -1)))
  (.gcd (biginteger 28) (biginteger 21))
  ;917
  (defn reverse-only-letters [s]
    (let [cs (char-array s)
          letter? #(and (>= (int %) (int \A)) (<= (int %) (int \z)))]
      (letfn [(swap-chars' [i j]
                (let [a (aget cs i) b (aget cs j)]
                  (aset cs i b)
                  (aset cs j a)
                  (swap-chars (inc i) (dec j))))
              (swap-chars [i j]
                (cond
                  (>= i j) true
                  (not (letter? (aget cs i))) (swap-chars (inc i) j)
                  (not (letter? (aget cs j))) (swap-chars i (dec j))
                  :else (swap-chars' i j)))]
        (swap-chars 0 (dec (count s)))
        (str/join "" (into [] cs)))))

;(map reverse-only-letters ["ab-cd" "a-bC-dEf-ghIj" "Test1ng-Leet=code-Q!"])
  (defn sort-array-by-partity-ii [nums]
    (flatten (map vector (filter even? nums) (filter odd? nums))))
                                        ;(map sort-array-by-partity-ii [[4 2 5 7] [2 3]])
  ;925
  (defn is-long-pressed-name [name typed]
    (letfn [(add [cs [results letters] i]
              (cond
                (= i (count cs)) (if (pos? (count letters))
                                   [(conj results letters) []]
                                   [results []])
                (= (cs (dec i)) (cs i)) [results (conj letters (cs i))]
                :else [(conj results letters) [(cs i)]]))
            (group-letters [s]
              (let [cs (vec s)
                    rs (range 1 (inc (count s)))]
                (first (reduce #(add cs %1 %2) [[] [(cs 0)]] rs))))]
      (let [css1 (group-letters name)
            css2 (group-letters typed)
            long-pressed? (fn [i]
                            (let [cs1 (css1 i)
                                  cs2 (css2 i)]
                              (and (= (cs1 0) (cs2 0)) (<= (count cs1) (count cs2)))))]
        (cond
          (= name typed) true
          (not= (count css1) (count css2)) false
          :else (every? long-pressed? (range (count css1)))))))

                                        ;(map (partial apply is-long-pressed-name) ['("alex" "aaleex") '("saeed" "ssaaedd") '("leelee" "lleeelee") '("laiden" "laiden")])
  ;929
  (defn num-unique-emails [emails]
    (letfn [(normalize [local-name]
              (->> (str/split local-name #"\+")
                   (first)
                   (#(str/replace % "." ""))))
            (unique-email [email]
              (let [[local-name domain-name] (str/split email #"@")]
                (str (normalize local-name) "@" domain-name)))]
      (count (set (map unique-email emails)))))

;; (map num-unique-emails [["test.email+alex@leetcode.com","test.e.mail+bob.cathy@leetcode.com","testemail+david@lee.tcode.com"]
  ;; ["a@leetcode.com","b@leetcode.com","c@leetcode.com"]])
  ;937
  (defn recorder-log-files [logs]
    (let [letter-log? (fn [log] (Character/isLetter (last (seq log))))
          digit-log? (fn [log] (Character/isDigit (last (seq log))))
          compare-content-then-identifier (fn [log1 log2]
                                            (if (= (rest log1) (rest log2))
                                              (compare (first log1) (first log2))
                                              (compare (str (rest log1)) (str (rest log2)))))
          letter-logs  (->> logs
                            (filter letter-log?)
                            (map #(str/split % #"\s+"))
                            (mapv #(cons (first %) (sort (rest %))))
                            (sort compare-content-then-identifier)
                            (map #(str/join " " %)))
          digit-logs (filter digit-log? logs)]
      (concat letter-logs digit-logs)))
  ; (map recorder-log-files [["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"]
  ;                          ["a1 9 2 3 1","g1 act car","zo4 4 7","ab1 off key dog","a8 act zoo"]])
  ;941
  (defn valid-moutain-array [arr]
    (let [max-value (apply max arr)
          index (.indexOf arr max-value)
          xs (subvec arr 0 (inc index))
          ys (subvec arr index (count arr))
          increasing? (fn [coll] (every? #(< (coll (dec %)) (coll %)) (range 1 (count coll))))
          decreasing? (fn [coll] (every? #(> (coll (dec %)) (coll %)) (range 1 (count coll))))]
      (if (or (< (count xs) 2) (< (count ys) 2))
        false
        (and (increasing? xs) (decreasing? ys)))))
  ;(map valid-moutain-array [[2 1] [3 5 5] [0 3 2 1]])
  ;942
  (defn di-string-match [s]
    (let [cs (vec s)
          add (fn [[result i j] index]
                (cond
                  (= index (count s)) [(conj result i) i j]
                  (= \I (cs index)) [(conj result i) (inc i) j]
                  :else [(conj result j) i (dec j)]))]

      (first (reduce add [[] 0 (count s)] (range (inc (count s)))))))
  (map di-string-match ["IDID" "III" "DDI"])
  ;944
  (defn min-deletion-size [strs]
    (letfn [(transpose [m]
              (let [rows (count m)
                    cols (count (m 0))
                    mt (make-array Character cols rows)]
                (doseq [i (range cols) j (range rows)]
                  (aset mt i j ((m j) i)))
                (mapv vec mt)))
            (non-increasing? [cs]
              (if (< (count cs) 2)
                true
                (not (every? #(< (int (cs (dec %))) (int (cs %))) (range 1 (count cs))))))]

      (->> strs
           (mapv vec)
           (transpose)
           (filter non-increasing?)
           (count))))
  ;(map min-deletion-size [["cba" "daf" "ghi"] ["a" "b"] ["zyx" "wvu" "tsr"]])
  ;953
  (defn is-alien-sorted [words order]
    (let [letters (vec order)
          english-letters (vec "abcdefghijklmnopqrstuvwxyz")
          translate (fn [word] (->> (vec word)
                                    (map #(english-letters (.indexOf letters %)))
                                    (str/join "")))
          increasing? (fn [ws]
                        (every? #(not= -1 (compare (ws (dec %)) (ws %)))
                                (range 1 (count ws))))]
      (->> words
           (mapv translate)
           (increasing?))))
  ;; (map (partial apply is-alien-sorted) ['(["hello","leetcode"] "hlabcdefgijkmnopqrstuvwxyz")
  ;;                                       '(["word","world","row"] "worldabcefghijkmnpqstuvxyz")
  ;;                                       '(["apple","app"] "abcdefghijklmnopqrstuvwxyz")])
  ;961
  (defn repeated-n-times [nums]
    (let [num-map {}
          add (fn [[num-map result] num]
                (if (nil? (get num-map num))
                  [(assoc num-map num true) []]
                  (reduced [num-map [num]])))]
      (->> nums
           (reduce add [{} []])
           (last)
           (last))))
                                 ;(map repeated-n-times [[1 2 3 3] [2 1 2 5 3 2] [5 1 5 2 5 3 5 4]])
  ;976
  (defn largest-perimeter [nums]
    (let [len (count nums)
          xs (vec (sort nums))
          triangles (for [i (range 0 (- len 2))
                          j (range (inc i) (- len 1))
                          k (range (inc j) len)]
                      [(xs i) (xs j) (xs k)])
          triangle? (fn [[i j k]]
                      (> (+ i j) k))
          max-perimeter (fn [result t] (if (triangle? t)
                                         (max result (apply + t))
                                         result))]
      (reduce max-perimeter 0 triangles)))

  ;(map largest-perimeter [[2 1 2] [1 2 1] [3 2 3 4] [3 6 2 3]])
                                        ;977
  (defn sorted-squares [nums]
    (let [squares (long-array (count nums))
          abs #(int (.abs (biginteger %1)))
          insert (fn [[l r] i]
                   (let [a (abs (nums l))
                         b (abs (nums r))]
                     (aset squares i (max (* a a) (* b b)))
                     (if (> a b)
                       [(inc l) r]
                       [l (dec r)])))
          rs (reverse (range (count nums)))]
      (reduce insert [0 (dec (count nums))] rs)
      (vec squares)))
  (map sorted-squares [[-4,-1,0,3,10] [-7,-3,2,3,11]])
                                        ;989
  (defn add-to-array-form [num k]
    (letfn [(integer-digits' [n]
              (let [n' (quot n 10)
                    digit (rem n 10)]
                (if (zero? n')
                  [digit]
                  (conj (integer-digits' n') digit))))
            (integer-digits [n]
              (if (zero? n)
                [0]
                (integer-digits' n)))]

      (let [xs (vec (reverse (vec num)))
            ys (vec (reverse (integer-digits k)))
            max-len (max (count xs) (count ys))
            min-len (min (count xs) (count ys))
            get-value (fn [coll index]
                        (if (< index (count coll))
                          (coll index)
                          0))
            add? (fn [carry i]
                   (or (< i max-len)
                       (pos? carry)))
            add (fn [[result carry] i]
                  (let [s (+ (get-value xs i) (get-value ys i) carry)]
                    (if (add? carry i)
                      [(cons (rem s 10) result) (quot s 10)]
                      [result 0])))]

        (reduce add [[] 0] (range (inc max-len))))))

  ;(map (partial apply add-to-array-form) ['([1 2 0 0] 34) '([2 7 4] 181) '([2 1 5] 806) '([9 9 9 9 9 9 9 9 9 9] 1)])
                                        ;997
  (defn find-judge [n trust]
    (let [find-judge' (fn [n trust]
                        (let [add (fn [result [a b]]
                                    (let [persons (get result b)]
                                      (if (empty? persons)
                                        (assoc result b [a])
                                        (assoc result b (conj persons a)))))
                              trust-map (reduce add {} trust)
                              non-judges (set (map first trust))
                              trust-nobody? (fn [k]
                                              (not (contains? non-judges k)))
                              trust-by-everybody? (fn [k persons]
                                                    (= (sort (conj persons k)) (range 1 (inc n))))
                              judge? (fn [[k persons]]
                                       (and (trust-nobody? k) (trust-by-everybody? k persons)))]
                          (->> trust-map
                               (filter judge?)
                               (first)
                               (#(if (empty? %) -1 (first %))))))]
      (if (< (count trust) (- n 1))
        -1
        (find-judge' n trust))))
  ;(map (partial apply find-judge) ['(2 [[1 2]]) '(3 [[1 3] [2 3]]) '(3 [[1 3] [2 3] [3 1]]) '(3 [[1 2] [2 3]]) '(4 [[1 3] [1 4] [2 3] [2 4] [4 3]])])
  ;999
  (defn num-rook-captures [board]
    (let [rows (count board)
          cols (count (board 0))
          positions (for [i (range rows)
                          j (range cols)]
                      [i j])
          find-rook (fn [result [i j]]
                      (if (= "R" ((board i) j))
                        (reduced [i j])
                        result))
          [starty startx] (reduce find-rook [] positions)
          directions [[-1 0] [1 0] [0 -1] [0 1]]
          valid-position? (fn [y x] (and (>= x 0) (< x 8) (>= y 0) (< y 8)))
          attack-pawn? (fn [[dy dx] result i]
                         (let [y (+ starty (* dy i))
                               x (+ startx (* dx i))
                               piece #((board y) x)]
                           (cond
                             (not (valid-position? y x)) (reduced false)
                             (= "p" (piece)) (reduced true)
                             (= "B" (piece)) (reduced false)
                             :else false)))
          search-pawn (fn [[dy dx]]
                        (reduce #(attack-pawn? [dy dx] %1 %2)
                                false
                                (range 1 8)))]

      (->> (map search-pawn directions)
           (filter true?)
           (count))))

;; (map num-rook-captures [[["." "." "." "." "." "." "." "."] ["." "." "." "p" "." "." "." "."] ["." "." "." "R" "." "." "." "p"] ["." "." "." "." "." "." "." "."] ["." "." "." "." "." "." "." "."] ["." "." "." "p" "." "." "." "."] ["." "." "." "." "." "." "." "."] ["." "." "." "." "." "." "." "."]]
  ;;                         [["." "." "." "." "." "." "." "."] ["." "p" "p" "p" "p" "p" "." "."] ["." "p" "p" "B" "p" "p" "." "."] ["." "p" "B" "R" "B" "p" "." "."] ["." "p" "p" "B" "p" "p" "." "."] ["." "p" "p" "p" "p" "p" "." "."] ["." "." "." "." "." "." "." "."] ["." "." "." "." "." "." "." "."]]
  ;;                         [["." "." "." "." "." "." "." "."] ["." "." "." "p" "." "." "." "."] ["." "." "." "p" "." "." "." "."] ["p" "p" "." "R" "." "p" "B" "."] ["." "." "." "." "." "." "." "."] ["." "." "." "B" "." "." "." "."] ["." "." "." "p" "." "." "." "."] ["." "." "." "." "." "." "." "."]]])
;1002
  (defn common-chars [words]
    (let [chars (apply clojure.set/intersection (map (comp set seq) words))
          freqs (map frequencies words)
          common-chars' (fn [c]
                          (let [counts (map #(get % c) freqs)
                                len (apply min counts)]
                            (take len (repeat c))))]
      (reduce concat [] (map common-chars' chars)))))

  ;(map common-chars [["bella" "label" "roller"] ["cool" "lock" "cook"]])
  ;; (defn win []
  ;;   (let [x (Integer/parseInt (read-line))]
  ;;     (println x)))
  ;; (win)
;)
(defn l-main3
  [& args]
  (defn cross-river [left right]
    (letfn [(cross-river' [left right [i j]]
              (let [a (left i)
                    b (left j)
                    left-duration (max a b)
                    remove-element (fn [xs elem]
                                     (let [index (.indexOf xs elem)]
                                       (into (subvec xs 0 index) (subvec xs (inc index)))))
                    left' #(conj (remove-element (remove-element left a) b) %)
                    right' (fn [elem] (remove-element (into [a b] right) elem))
                    get-duration (fn [duration] (+ duration (cross-river (left' duration) (right' duration))))
                    right-duration (apply min (map get-duration (into [a b] right)))]
                (println (str "l = " left " r = " right))
                (+ left-duration right-duration)))]
      (let [len (count left)
            lefts (for [i (range (dec len)) j (range (inc i) len)] [i j])]
        (if (<= len 2)
          (apply max left)
          (apply min (map (partial cross-river' left right) lefts)))))))

;; (println (cross-river [1 2 4 8] []))
(comment
(defn largest-sum-after-k-negations [nums k]
  (let [total #(apply + %)
        negative (fn [result _]
                   (sort (cons (* -1 (first result)) (rest result))))]
    (total (reduce negative (sort nums) (range k)))))
(map (partial apply largest-sum-after-k-negations) ['([4 2 3] 1) '([3 -1 0 2] 3)])

;1009
(defn bitwise-complement [n]
  (letfn [(bitwise-complement' [n]
            (if (zero? n)
              0
              (+ (bit-shift-left (bitwise-complement' (bit-shift-right n 1)) 1)
                 (bit-and (bit-xor n 1) 1))))]
    (if (zero? n)
      1
      (bitwise-complement' n))
    ;  (bit-shift-right (bit-not n) 1)
    ))
(map bitwise-complement [5 7 10 0])

;1013
(defn can-three-parts-equal-sum [arr]
  (let [total (apply + arr)
        num-map (frequencies arr)
        can-equal-sum? (fn [[s parts] x]
                         (if (= (+ s x) (quot (* (inc parts) total) 3))
                           [(+ s x) (inc parts)]
                           [(+ s x) parts]))
        can-three-parts-equal-sum' #(>= (last (reduce can-equal-sum? [0 0] %)) 3)]
    (if (zero? (rem total 3))
      (can-three-parts-equal-sum' arr)
      false)))
(map can-three-parts-equal-sum [[0 2 1 -6 6 -7 9 1 2 0 1] [0 2 1 -6 6 7 9 -1 2 0 1] [3 3 6 5 -2 2 5 1 -9 4]])
;1018
(defn prefixes-div-by-5 [nums]
  (let [sum (fn [results num]
              (+ (* (last results) 2)
                 num))
        to-integer (fn [results num]
                     (if (empty? results)
                       [num]
                       (conj results (sum results num))))
        div-by-5? (fn [x] (zero? (rem x 5)))]
    (map div-by-5? (reduce to-integer [] nums))))
(map prefixes-div-by-5 [[0 1 1] [1 1 1] [0 1 1 1 1 1] [1 1 1 0 1]])

(defn remove-outer-parentheses1 [s]
  (let [matched? (fn [xs]
                   (let [m (frequencies xs)]
                     (and (= (count m) 2)
                          (= (count (set (vals m))) 1))))
        add (fn [[results result] c]
              (let [new-result (conj result c)]
                (if (and (= c \) ) (matched? new-result))
                  [(conj results new-result) []]
                  [results new-result])))]
    (->> (reduce add [[] []] (vec s))
         (first)
         (map #(rest (pop %)))
         (flatten)
         (#(str/join "" %)))))

;1021
(defn remove-outer-parentheses [s]
  (let [add (fn [[xs opens] c]
              (cond
                (= c \() (if (> opens 0)
                           [(conj xs c) (inc opens)]
                           [xs (inc opens)])
                :else (if (> opens 1)
                        [(conj xs c) (dec opens)]
                        [xs (dec opens)])))]
    (->> (reduce add [[] 0] (vec s))
         (first)
         (#(str/join "" %)))))

(map remove-outer-parentheses ["(()())(())" "(()())(())(()(()))" "()()"])

;1025
(defn divisor-game [n]
  (if (= n 1)
    false
    (not (divisor-game (dec n)))))
(map divisor-game [2 3 4])

;1030
(defn all-cells-dist-order [rows cols r-center c-center]
  (let [abs (fn [n] (if (pos? n) n (* -1 n)))
        distance (fn [r c] (+ (abs (- r r-center)) (abs (- c c-center))))
        cells (for [r (range rows) c (range cols)] [r c])
        distance-map (reduce (fn [m [r c]] (assoc m [r c] (distance r c))) {} cells)
        compare-value (fn [key1 key2] (compare [(get distance-map key1) key1] [(get distance-map key2) key2]))]
    (->> distance-map
         (into (sorted-map-by compare-value))
         (keys))))
(map (partial apply all-cells-dist-order) ['(1 2 0 0) '(2 2 0 1) '(2 3 1 2)])

;1037
(defn is-boomerang [points]
  (let [minus (fn [[x1 y1] [x2 y2]] [(- x1 x2) (- y1 y2)])
        [v1 v2] (minus (points 1) (points 0))
        [w1 w2] (minus (points 2) (points 0))]
    (not= (* v1 w2) (* v2 w1))))
(map is-boomerang [[[1 1] [2 3] [3 2]] [[1 1] [2 2] [3 3]]])

;1046
(defn last-stone-weight [stones]
  (letfn [(play [stones]
            (let [ss (sort > stones)
                  x (nth ss 1)
                  y (nth ss 0)]
              (if (= x y)
                (rest (pop ss))
                (conj (rest (drop-last ss)) (- y x)))))]
    (cond
      (empty? stones) 0
      (= 1 (count stones)) (first stones)
      :else (last-stone-weight (play stones)))))
(map last-stone-weight [[2 7 4 1 8 1] [1]])
;1047
(defn remove-duplicates1 [s]
  (letfn [(add [[result tmp] c]
            (cond
              (empty? tmp) [result [c]]
              (and (not= (last tmp) c) (> (count tmp) 1)) [result [c]]
              (and (not= (last tmp) c) (= (count tmp) 1)) [(concat result tmp) [c]]
              :else [result (conj tmp c)]))
          (remove-dups [cs]
            (let [rs (apply concat (reduce add [[] []] cs))]
              (if (= cs rs)
                (str/join "" cs)
                (remove-dups rs))))]
    (remove-dups (vec s))))

(defn remove-duplicates [s]
  (let [remove-dups (fn [result c]
                      (if (and (pos? (count result)) (= (last result) c))
                        (drop-last result)
                        (conj result c)))]
    (str/join "" (reduce remove-dups [] (vec s)))))
(map remove-duplicates ["abbaca" "azxxzy"])

(defn cross-river [left right]
  (letfn [(cross-river' [left right [i j]]
            (let [a (left i)
                  b (left j)
                  left-duration (max a b)
                  remove-element (fn [xs elem]
                                   (let [index (.indexOf xs elem)]
                                     (into (subvec xs 0 index) (subvec xs (inc index)))))
                  left' #(conj (remove-element (remove-element left a) b) %)
                  right' (fn [elem] (remove-element (into [a b] right) elem))
                  get-duration (fn [duration] (+ duration (cross-river (left' duration) (right' duration))))
                  right-duration (apply min (map get-duration (into [a b] right)))]
              (println (str "l = " left-duration " r = " right-duration))
              (+ left-duration right-duration)))]
    (let [len (count left)
          lefts (for [i (range (dec len)) j (range (inc i) len)] [i j])]
      (if (<= len 2)
        (apply max left)
        (apply min (map (partial cross-river' left right) lefts))))))

;(cross-river [1 2 4 8 16] [])
(defn height-checker [heights]
  (->> (map not= heights (sort heights))
       (filter true?)
       (count)))
(map height-checker [[1,1,4,2,1,3] [5,1,2,3,4] [1,2,3,4,5]])

;1071
(defn gcd-of-strings1 [str1 str2]
  (let [rows (count str1)
        cols (count str2)
        cs1 (vec str1)
        cs2 (vec str2)
        dp (make-array Long/TYPE (inc rows) (inc cols))
        cells (for [r (range rows) c (range cols)] [r c])
        max-common-str-len (fn [[len col] [r c]]
                             (if (< len (aget dp (inc r) (inc c)))
                               [(aget dp (inc r) (inc c)) c]
                               [len col]))]
    (doseq [[r c] cells]
      (when (= (cs1 r) (cs2 c))
        (aset dp (inc r) (inc c) (inc (aget dp r c)))))
    (let [[len col] (reduce max-common-str-len [0 nil] cells)]
      (if (zero? len)
        ""
        (str/join "" (subvec cs2 (- (inc col) len) (inc col)))))))

(defn gcd-of-strings2 [str1 str2]
  (letfn [(gcd [n1 n2]
            (let [[a b] (if (> n1 n2) [n1 n2] [n2 n1])]
              (if (zero? (rem a b))
                b
                (gcd b (rem a b)))))
          (common-divisor? [s divisor]
            (let [compare-string (fn [result i]
                                   (let [len (count divisor)
                                         part (subs s (* i len) (* (inc i) len))]
                                     (if (= part divisor)
                                       true
                                       (reduced false))))
                  len (quot (count s) (count divisor))]
              (reduce compare-string true (range len))))]

    (let [len (gcd (count str1) (count str2))
          divisor (subs str1 0 len)]
      (and (common-divisor? str1 divisor) (common-divisor? str2 divisor)))))

(defn gcd-of-strings [str1 str2]
  (letfn [(gcd [a' b']
            (let [[a b] [(max a' b') (min a' b')]]
              (if (zero? (rem a b))
                b
                (gcd b (rem a b)))))]
    (if (= (str str1 str2) (str str2 str1))
      (subs str1 0 (gcd (count str1) (count str2)))
      "")))

(map (partial apply gcd-of-strings) ['("ABCABC" "ABC") '("ABABAB" "ABAB") '("LEET" "CODE") '("ABCDEF" "ABC")])

;1078
(defn find-occurrences [text fs ss]
  (let [ws (vec (re-seq #"\w+" text))
        first-second-matched? (fn [i]
                                (if (> i 1)
                                  (and (= (ws (- i 2)) fs)
                                       (= (ws (dec i))))
                                  false))
        indices (filter first-second-matched? (range (count ws)))]
    (map #(ws %) indices)))

(map (partial apply find-occurrences) ['("alice is a good girl she is a good student" "a" "good") '("we will we will rock you" "we" "will")])

;1089
(defn duplicate-zeros1 [arr]
  (let [indices (filter #(zero? (get arr %)) (range (count arr)))
        insert-zero (fn [[result start] index]
                      [(concat result (subvec arr start (inc index)) [0]) (inc index)])
        [result start] (reduce insert-zero [[] 0] indices)
        duplicate-zeroes' (fn [xs]
                            (let [len (count xs)
                                  ys (concat result (subvec arr start len))]
                              (vec (take len ys))))]
    (if (empty? indices)
      arr
      (duplicate-zeroes' arr))))

(defn duplicate-zeros [arr]
  (let [xs (into-array arr)]
    (letfn [(dup [coll i j]
              (let [dup-zero (fn [coll i j]
                               (aset coll j (arr i))
                               (aset coll (inc j) 0)
                               (dup coll (inc i) (+ j 2)))
                    dup-non-zero (fn [coll i j]
                                   (aset coll j (arr i))
                                   (dup coll (inc i) (inc j)))]
                (cond
                  (>= j (count arr)) coll
                  (zero? (arr i)) (dup-zero coll i j)
                  :else (dup-non-zero coll i j))))]
      (into [] (dup xs 0 0)))))

(map duplicate-zeros [[1 0 2 3 0 4 5 0] [1 2 3]])
;1103
(defn distribute-candies1 [candies num-people]
  (let [s1 (quot (* num-people (inc num-people)) 2)
        distribute1 #(reduce (fn [[xs s] i]
                               (if (> (+ s i) candies)
                                 (reduced [(conj xs (- candies s)) candies])
                                 [(conj xs i) (+ s i)]))
                             [[] 0]
                             (range 1 (inc num-people)))
        distribute2 #(reduce (fn [[xs s] i]
                               (if (> (+ s i) candies)
                                 (reduced [(vec (concat (conj xs (+ (- i num-people) (- candies s)))
                                                        (range (- (inc i) num-people)
                                                               (inc num-people))))
                                           candies])
                                 [(conj xs (- (* 2 i) num-people)) (+ s i)]))
                             [[] s1]
                             (range (inc num-people) (inc (* 2 num-people))))]
    (first (if (<= candies s1)
             (distribute1)
             (distribute2)))))

(defn distribute-candies [candies num-people]
  (let [cs (make-array Long/TYPE num-people)]
    (letfn [(distribute [xs candies give]
              (let [index (rem (inc give) num-people)
                    value (+ (aget xs index) (min candies (inc give)))
                    next-give (inc give)]
                (aset xs index value)
                (when (> candies next-give)
                  (distribute xs (- candies next-give) next-give))
                xs))]
      (into [] (distribute cs candies 0)))))
(map (partial apply distribute-candies) ['(7 4) '(10 3)])
;1108
(defn defang-ip-addr [address]
  ;; (->> (str/split address #"[.]")
  ;;      (str/join "[.]")
  ;;      )
  (str/replace address "." "[.]"))
(map defang-ip-addr ["1.1.1.1" "255.100.50.0"])

;1122
(defn relative-sort-array [arr1 arr2]
  (let [freqs (frequencies arr1)
        distinct-numbers (flatten (map #(take (get freqs %) (repeat %)) arr2))
        different-numbers (sort (clojure.set/difference (set arr1) (set arr2)))]
    (concat distinct-numbers different-numbers)))
(map (partial apply relative-sort-array) ['([2,3,1,3,2,4,6,7,9,2,19] [2,1,4,3,9,6]) '([28,6,22,8,44,17] [22,28,8,6])])

;1128
(defn num-equiv-domino-pairs [dominoes]
  (->> (map sort dominoes)
       (frequencies)
       (vals)
       (#(apply max %))
       (#(quot (* % (dec %)) 2))))
(map num-equiv-domino-pairs [[[1,2],[2,1],[3,4],[5,6]] [[1,2],[1,2],[1,1],[1,2],[2,2]]])
;;1137
(defn tribonacci [n]
  (letfn [(add [[a b c] _]
            [b c (+ a b c)])
          (tribonacci' [a b c n' n]
            (last (reduce add [a b c] (range n' (inc n)))))]
    (or (get {0 0 1 1 2 1} n)
        (tribonacci' 0 1 1 3 n))))
(map tribonacci [4 25])

(defn day-of-year [date]
  (let [month-days [31 28 31 30 31 30 31 31 30 31 30 31]
        leap-year? (fn [y] (or (and (zero? (rem y 4)) (not= 0 (rem y 100))) (zero? (rem y 400))))
        [year month day] (map #(Integer/parseInt %) (str/split date #"-"))
        extra-day (if (and (leap-year? year) (> month 2)) 1 0)
        part1 (apply + (subvec month-days 0 (dec month)))
        part2 day]

    (+ extra-day
       part1
       part2)))
(map day-of-year ["2019-01-09" "2019-02-10" "2003-03-01" "2004-03-01"])
;;1160
(defn count-characters [words chars]
  (let [char-map (frequencies chars)
        good? (fn [[c num]] (>= (or (char-map c) 0) num))
        freq? (fn [s] (every? good? (frequencies (seq s))))]
    (->> (filter freq? words)
         (map count)
         (apply +))))
(map (partial apply count-characters) ['(["cat" "bt" "hat" "tree"] "atach") '(["hello" "world" "leetcode"] "welldonehoneyr")])
;;1184
(defn distance-between-bus-stops1 [distance start destination]
  (let [len (count distance)
        [src dst] (if (< start destination) [start destination] [destination start])
        get-distance #(get distance (rem % len))
        get-total-distance (fn [s d] (apply + (map get-distance (range s d))))]
    (min (get-total-distance src dst)
         (get-total-distance dst (+ src len)))))

(defn distance-between-bus-stops [distance start destination]
  (let [add (fn [[sum1 sum2] i]
              (if (and (<= start i) (< i destination))
                [(+ sum1 (distance i)) sum2]
                [sum1 (+ sum2 (distance i))]))]
    (apply min (reduce add [0 0] (range (count distance))))))

(map (partial apply distance-between-bus-stops) ['([1,2,3,4] 0 1) '([1 2 3 4] 0 2) '([1 2 3 4] 0 3)])
;;1185
(defn day-of-the-week [day month year]
  ;;1971-01-01 Friday
  (let [dates ["Sunday" "Monday" "Tuesday" "Wednesday" "Thursday" "Friday" "Saturday"]
        month-days [31 28 31 30 31 30 31 31 30 31 30 31]
        leap-year? (fn [year] (or (and (zero? (rem year 4))
                                       (not= 0 (rem year 100)))
                                  (zero? (rem year 400))))
        days (+ (apply + (map #(if (leap-year? %) 366 365) (range 1971 year)))
                (+ (apply + (map #(month-days (dec %1)) (range 1 month)))
                   day
                   (if (and (leap-year? year) (> 2 month)) 1 0)))
        index (rem (+ 4 (rem days 7)) 7)]
    (dates index)))
(map (partial apply day-of-the-week) ['(31 8 2019) '(18 7 1999) '(15 8 1993)])

;;1189
(defn max-number-of-ballons [text]
  (let [ballon-freqs (frequencies (seq "balloon"))
        word-freqs (frequencies text)
        get-letter-count #(or (get word-freqs %) 0)]
    (apply min (map (fn [[letter num]]
                      (quot (get-letter-count letter) num))
                    ballon-freqs))))
(map max-number-of-ballons ["nlaebolko" "loonbalxballpoon" "leetcode"])
;;1200
(defn mininum-abs-difference [arr]
  (let [xs (vec (sort arr))
        add (fn [result i]
              (let [v1 (xs (dec i))
                    v2 (xs i)
                    d (- v2 v1)
                    value (conj (or (get result d) []) [v1 v2])]
                (assoc result d value)))
        difference-map (reduce add {} (range 1 (count xs)))
        key (apply min (keys difference-map))]
    (difference-map key)))
(map mininum-abs-difference [[4 2 1 3] [1 3 6 10 15] [3 8 -10 23 19 -4 -14 27]])

;;1207
(defn unique-occurences [arr]
  (let [freqs (frequencies arr)]
    (= (count (keys freqs)) (count (distinct (vals freqs))))))
(map unique-occurences [[1 2 2 1 1 3] [1 2] [-3 0 1 -3 1 1 1 -3 10 0]])

;;1217
(defn min-cost-to-move-chips [position]
  (min (count (filter even? position))
       (count (filter odd? position))))
(map min-cost-to-move-chips [[1 2 3] [2 2 2 3 3] [1 1000000000]])

;;1221
(defn balanced-string-split [s]
  (let [split-string (fn [[results result opened] c]
                       (cond
                         (and (= c \L) (zero? (dec opened))) [(conj results (conj result c)) [] 0]
                         (and (= c \L) (pos? (dec opened))) [results (conj result c) (dec opened)]
                         :else [results (conj result c) (if (= \R c) (inc opened) (dec opened))]))
        balanced-string-split' (fn [coll] (first (reduce split-string [[] [] 0] coll)))]
    (balanced-string-split' (vec s))))
(map balanced-string-split ["RLRRLLRLRL" "RLLLLRRRLR" "LLLLRRRR" "RLRRRLLRLL"])

(defn check-straight-line [coordinates]
  (let [minus (fn [[x1 y1] [x2 y2]] [(- x1 x2) (- y1 y2)])
        straight-line? (fn [[v1 v2] [w1 w2]] (= (* v1 w2) (* v2 w1)))
        [x1 y1] (first coordinates)
        vectors (reduce (fn [results [x y]] (conj results (minus [x1 y1] [x y]))) [] (rest coordinates))
        v (first vectors)
        check-straight-line' (fn [vectors] (every? (fn [w] (straight-line? v w)) (rest vectors)))]
    (if (= 2 (count coordinates))
      true
      (check-straight-line' vectors))))
(map check-straight-line [[[1 2] [2 3] [3 4] [4 5] [5 6] [6 7]] [[1 1] [2 2] [3 4] [4 5] [5 6] [7 7]]])
;
(defn shift-grid [grid k]
  (let [rows (count grid)
        cols (count (get grid 0))]
    (letfn [(shift [g _]
              (let [m (make-array Long/TYPE rows cols)
                    cells (for [r (range rows) c (range (dec cols))] [r c])]
                (doall (map (fn [[r c]] (aset m r (inc c) ((g r) c))) cells))
                (doseq [r (range (dec rows))]
                  (aset m (inc r) 0 ((g r) (dec cols))))
                (aset m 0 0 ((g (dec rows)) (dec cols)))
                (mapv vec m)))]
      (reduce shift grid (range k)))))
(map (partial apply shift-grid) ['([[1 2 3] [4 5 6] [7 8 9]] 1) '([[3 8 1 9] [19 7 2 5] [4 6 11 10] [12 0 21 13]] 4) '([[1 2 3] [4 5 6] [7 8 9]] 9)])
;;1266
(defn min-time-to-visit-all-points [points]
  (letfn [(abs [a] (if (neg? a) (* -1 a) a))
          (time [[x1 y1] [x2 y2]]
            (max (abs (- x1 x2)) (abs (- y1 y2))))]
    (reduce (fn [s i]
              (+ s (time (points i) (points (dec i))))) 0 (range 1 (count points)))))
(map min-time-to-visit-all-points [[[1 1] [3 4] [-1 0]] [[3 2] [-2 2]]])

;;1275
(defn tictactoe [moves]
  (let [rows 3
        cols 3
        b (make-array Long/TYPE rows cols)
        setup-board (fn [moves] (do
                                  (doseq [i (range (count moves))]
                                    (let [[r c] (get moves i)
                                          value (if (even? i) -1 1)]
                                      (aset b r c value)))
                                  b))
        win (fn [board]
              (let [win-in-row? (fn [r]
                                  (reduce (fn [result c] (if (= (aget board r c) (aget board r 0))
                                                           true
                                                           (reduced false)))
                                          true (range 1 cols)))
                    row-winner
                    (reduce (fn [_ i] (if (win-in-row? i)
                                        (reduced (aget board i 0))
                                        nil)) nil (range rows))
                    win-in-col? (fn [c]
                                  (reduce (fn [result r] (if (= (aget board r c) (aget board 0 c))
                                                           true
                                                           (reduced false)))
                                          true (range 1 rows)))

                    col-winner
                    (reduce (fn [_ c] (if (win-in-col? c)
                                        (reduced (aget board 0 c))
                                        nil)) nil (range cols))
                    to-winner (fn [w]
                                (cond
                                  (= w -1) "A"
                                  (= w 1) "B"
                                  :else nil))

                    win-in-diagnal? (fn [] (reduce (fn [result r] (if (= (aget board r r) (aget board 0 0))
                                                                    true
                                                                    (reduced false))) false (range rows)))
                    win-in-reverse-diagnal? (fn []
                                              (reduce (fn [result r] (if (= (aget board 0 (dec cols)) (aget board r (- (dec rows) r)))
                                                                       true
                                                                       (reduced false))) false (range rows)))
                    diagonal-winner (if (or (win-in-diagnal?) (win-in-reverse-diagnal?))
                                      (aget board (quot rows 2) (quot cols 2))
                                      nil)]
                (to-winner (or row-winner col-winner diagonal-winner))))

        pending? (fn [board]
                   (let [indices (for [r (range rows) c (range cols)] [r c])]
                     (pos? (count (filter (fn [[r c]] (zero? (aget board r c))) indices)))))]
    (let [board (setup-board moves)]
      (let [winner (win board)]
        (cond
          (not (nil?  winner)) winner
          (pending? board) "Pending"
          :else "Draw")))))

(map tictactoe [[[0 0] [2 0] [1 1] [2 1] [2 2]] [[0 0] [1 1] [0 1] [0 2] [1 0] [2 0]]
                [[0 0] [1 1] [2 0] [1 0] [1 2] [2 1] [0 1] [0 2] [2 2]]
                [[0 0] [1 1]]])

;;1281
(defn subtract-product-and-sum [n]
  (letfn [(integer-digits [n]
            (cond
              (zero? n) []
              (< n 10) [n]
              :else (conj (integer-digits (quot n 10)) (rem n 10))))]
    (let [digits (integer-digits n)]
      (- (apply * digits) (apply + digits)))))
(map subtract-product-and-sum [234 4421])

;;1287
(defn find-special-integer [arr]
  (let [n (count arr)
        count-freqs (fn [[m _] x] (let [num (inc (or (get m x) 0))]
                                    (if (> (/ num n) 0.25)
                                      (reduced [m x])
                                      [(assoc m x num) nil])))]
    (last
      (reduce count-freqs [{} nil] arr))))
(map find-special-integer [[1 2 2 6 6 6 6 7 10] [1 1]])
;;1295
(defn find-numbers [nums]
  (letfn [(integer-digits-length [n]
            (if (< n 10)
              1
              (inc (integer-digits-length (quot n 10)))))]
    (let [m (frequencies nums)]
      (->> (filter #(even? (integer-digits-length %)) (keys m))
           (map #(get m %))
           (apply +)))))
(map find-numbers [[12 345 2 6 7896] [555 901 482 1771]])
;;1299
(defn replace-elements [arr]
  (let [replace-element (fn [[result max-element] element]
                          (if (> element max-element)
                            [(cons element result) element]
                            [(cons max-element result) max-element]))]
    (first (reduce replace-element [[-1] (last arr)] (reverse (rest arr))))))
(map replace-elements [[17 18 5 4 6 1] [400]])
;;1304
(defn sum-zero1 [n]
  (let [result (vec (reduce (fn [result x] (concat result [x (* -1 x)])) [] (range 1 (inc (quot n 2)))))]
    (if (even? n)
      result
      (conj result 0))))

(defn sum-zero [n]
  (let [xs (range (dec n))
        negative-sum (* -1 (apply + xs))]
    (conj xs negative-sum)))
(map sum-zero [5 3 1])
;;1309
(defn freq-alphabets1 [s]
  (let [table (vec "abcdefghijklmnopqrstuvwxyz")
        decode (fn [letters]
                 (let [index (Integer/parseInt (str/join "" letters))]
                   (table (dec index))))]

    (letfn [(translate [cs i]
              (cond
                (neg? i) []
                (= \# (cs i)) (conj (translate cs (- i 3)) (decode (subvec cs (- i 2) i)))
                :else (conj (translate cs (dec i)) (decode [(cs i)]))))]
      (str/join "" (translate (vec s) (dec (count s)))))))

(defn freq-alphabets [s]
  (let [decode-wide-digits #(str (char (+ (Integer/parseInt (subs %1 0 2)) (int \j) -10)))
        decode-single-digit #(str (char (+ (Integer/parseInt %1) (int \a))))]
    (->> (str/replace s #"[12][0-9]#" decode-wide-digits)
         (#(str/replace % #"[0-9]" decode-single-digit)))))
(map freq-alphabets ["10#11#12" "1326#" "25#" "12345678910#11#12#13#14#15#16#17#18#19#20#21#22#23#24#25#26#"])
;;1313
(defn decompress-RLE-list [nums]
  (let [decompress (fn [i] (repeat (nums i) (nums (inc i))))
        rs (range 0 (count nums) 2)]
    (flatten (map  decompress rs))))
(map decompress-RLE-list [[1 2 3 4] [1 1 2 3]])
;;1317
(defn get-no-zero-integers [n]
  (letfn [(no-zero? [n]
            (cond
              (zero? n) false
              (< n 10) true
              (not (zero? (rem n 10))) (no-zero? (quot n 10))
              :else false))]
    (reduce (fn [result i]
              (if (and (no-zero? i) (no-zero? (- n i)))
                (reduced [i (- n i)])
                [])) [] (range 1 (inc (quot n 2))))))
(map get-no-zero-integers [2 11 10000 69 1010])
;;1323
(defn maximum-69-number [num]
  (letfn [(integer-digits [n]
            (if (< n 10)
              [n]
              (conj (integer-digits (quot n 10)) (rem n 10))))]
    (let [ds (into-array (vec (integer-digits num)))
          index (.indexOf (into [] ds) 6)]
      (if (= index -1)
        num
        (do
          (aset ds index 9)
          (reduce #(+ (* %1 10) %2) 0 ds))))))
(map maximum-69-number [9669 9996 9999])
;;1331
(defn array-rank-transform [arr]

  (let [xs (sort (distinct arr))
        ts (map (fn [i x] [i x]) (range 1 (inc (count xs))) xs)
        ranks (reduce (fn [m [i elem]] (assoc m elem i)) {} ts)]
    (map #(get ranks %) arr)))
(map array-rank-transform [[40 10 20 30] [100 100 100] [37 12 28 9 100 56 80 5 12]])
;;1332
(defn remove-palindrome-sub [s]
  (cond
    (= "" s) 0
    (= (vec s) (vec (reverse (vec s)))) 1
    :else 2))
(map remove-palindrome-sub ["ababa" "abb" "baabb"])
;;1337
(defn k-weakest-rows [mat k]
  (->> (map (partial apply +) mat)
       (#(map (fn [i num] [i num]) (range (count %)) %))
       (sort (fn [l h] (compare (last l) (last h))))
       (take k)
       (map first)))
(map (partial apply k-weakest-rows) ['([[1 1 0 0 0]
                                        [1 1 1 1 0]
                                        [1 0 0 0 0]
                                        [1 1 0 0 0]
                                        [1 1 1 1 1]] 3)
                                     '([[1 0 0 0]
                                        [1 1 1 1]
                                        [1 0 0 0]
                                        [1 0 0 0]] 2)])
;;1342
(defn number-of-steps [num]
  (letfn [(number-of-steps' [n]
            (cond
              (zero? n) 0
              (even? n) (inc (number-of-steps' (quot n 2)))
              :else (inc (number-of-steps' (dec n)))))]

    (number-of-steps' num)))
(map number-of-steps [14 8 123])
;;1346
(defn check-if-exists [arr]
  (let [freqs (frequencies arr)
        xs (drop-last (sort (keys freqs)))
        check (fn [result x] (if (and (not= 0 x) (get freqs (* 2 x)))
                               (reduced true)
                               false))]

    (reduce check false xs)))
(map check-if-exists [[10 2 5 3] [7 1 14 11] [3 1 7 11]])

;;1351
(defn count-negatives [grid]
  (let [count-negatives' (fn [xs]
                           (reduce (fn [result x]
                                     (if (neg? x)
                                       (+ result 1)
                                       (reduced result))) 0 (reverse xs)))]
    (apply + (map count-negatives' grid))))
(map count-negatives [[[4 3 2 -1] [3 2 1 -1] [1 1 -1 -2] [-1 -1 -2 -3]] [[3 2] [1 0]] [[1 -1] [-1 -1]] [[-1]]])
;;1356
(defn sort-by-bits [arr]
  (letfn [(count-bits [n]
            (if (< n 2)
              n
              (+ (count-bits (quot n 2)) (rem n 2))))
          (compare-bits [l h]
            (compare (first l) (first h)))]

    (->> (map (fn [x] [(count-bits x) x]) arr)
         (sort compare-bits)
         (map last))))
(map sort-by-bits [[0 1 2 3 4 5 6 7 8] [1024 512 256 128 64 32 16 8 4 2 1] [10000 10000] [2 3 5 7 11 13 17 19] [10 100 1000 10000]])
;;1360
(defn days-between-date [date1 date2]
  (let [month-map [31 28 31 30 31 30 31 31 30 31 30 31]
        [s1 s2] (if (neg? (compare date1 date2)) [date1 date2] [date2 date1])
        to-date (fn [s] (map #(Integer/parseInt %) (str/split s #"-")))
        [y1 m1 d1] (to-date s1)
        [y2 m2 d2] (to-date s2)
        leap-year? (fn [year] (or (and (zero? (rem year 4)) (not= 0 (rem year 100))) (zero? (rem year 400))))
        to-days (fn [year month day]
                  (let [year-days (reduce #(+ %1 (if (leap-year? %2) 366 365)) 0 (range (inc 1970) year))
                        month-days (reduce #(+ %1 (get month-map %2)) 0 (range 0 (dec month)))
                        leap-day (if (and (leap-year? year) (> month 2)) 1 0)]
                    (+ year-days month-days day leap-day)))]
    (- (to-days y2 m2 d2) (to-days y1 m1 d1))))
(map (partial apply days-between-date) ['("2019-06-29" "2019-06-30") '("2020-01-15" "2019-12-31")])
;;1365
(defn smaller-numbers-than-current [nums]
  (let [freqs (frequencies nums)
        xs (sort (keys freqs))
        counter (vec (reduce (fn [result num]
                               (conj result (+ (last result) (freqs num)))) [0] xs))]
    (map #(counter (.indexOf xs %)) nums)))
(map smaller-numbers-than-current [[8 1 2 2 3] [6 5 4 8] [7 7 7 7]])
;;1370
(defn sort-string [s]
  (let [letter-map (frequencies (vec s))
        letters (sort (keys letter-map))]
    (letfn [(add [m c] (let [num (get m c)]
                         (if (= num 1)
                           (dissoc m c)
                           (assoc m c (dec num)))))
            (sort-string' [m cs]
              (let [new-letter-map (reduce add  m cs)]
                (if (empty? cs)
                  []
                  (concat cs (sort-string' new-letter-map (reverse (keys new-letter-map)))))))]

      (sort-string' letter-map letters))))
(map sort-string ["aaaabbbbcccc" "rat" "leetcode" "ggggggg" "spo"])
(defn generate-the-string [n]
  (let [letters (if (odd? n)
                  (take n (repeat "x"))
                  (concat ["x"] (take (dec n) (repeat "y"))))]
    (str/join "" letters)))
(map generate-the-string [4 2 7])
;;1380
(defn lucky-numbers [matrix]
  (let [rows (count matrix)
        cols (count (matrix 0))
        lucky-number (fn [m [row col]]
                       (let [minr (apply min (m row))
                             maxc (apply max (map (fn [r] ((m r) col)) (range rows)))
                             num ((m row) col)]
                         (if (= num minr maxc)
                           minr
                           nil)))
        indices (for [r (range rows) c (range cols)] [r c])]
    (->> (map #(lucky-number matrix %) indices)
         (filter (comp not nil?)))))
(map lucky-numbers [[[3 7 8] [9 11 13] [15 16 17]] [[1 10 4 2] [9 3 8 7] [15 16 17 12]] [[7 8] [1 2]] [[3 6] [7 1] [5 2] [4 8]]])

;;1385
(defn find-the-distance-value [arr1 arr2 d]
  (let [freqs (frequencies arr1)
        ys (keys (frequencies arr2))
        abs (fn [n] (if (pos? n) n (* -1 n)))
        valid? (fn [num] (every? #(> (abs (- num %)) d) ys))]
    (->> (filter valid? (keys freqs))
         (map (fn [num] (freqs num)))
         (apply +))))
(map (partial apply find-the-distance-value) ['([4 5 8] [10 9 1 8] 2) '([1 4 2 3] [-4 -3 6 10 20 30] 3) '([2 1 100 3] [-5 -2 10 -3 7] 6)])

;;1389
(defn create-target-array [nums index]
  (let [num-indices (map (fn [num i] [num i]) nums index)
        insert (fn [result [num i]]
                 (if (empty? result)
                   [num]
                   (vec (concat (subvec result 0 i) [num] (subvec result i (count result))))))]
    (reduce insert [] num-indices)))
(map (partial apply create-target-array) ['([0 1 2 3 4] [0 1 2 2 1])
                                          '([1 2 3 4 0]  [0 1 2 3 0])
                                          '([1] [0])])
;;1394
(defn find-lucky [arr]
  (let [fs (frequencies arr)]
    (reduce (fn [result num]
              (if (= num (get fs num))
                (reduced num)
                result))
            -1
            (keys fs))))
(map find-lucky [[2,2,3,4] [1,2,2,3,3,3] [2,2,2,3,3] [5] [7,7,7,7,7,7,7]])
;;1399
(defn count-largest-group [n]
  (letfn [(integer-digits [n]
            (if (< n 10)
              [n]
              (conj (integer-digits (quot n 10)) (rem n 10))))
          (integer-digits-sum [n]
            (apply + (integer-digits n)))]
    (let [xs (map (fn [x] [(integer-digits-sum x) x]) (range 1 (inc n)))
          num-counter (frequencies (map first xs))
          max-frequent (first (sort > (vals num-counter)))]
      (count (filter #(= (last %) max-frequent) (into [] num-counter))))))
(map count-largest-group [13 2 15 24])
;;1403
(defn min-subsequence [nums]
  (let [total (apply + nums)
        add (fn [[result s] num]
              (if (<= (+ s num) (quot total 2))
                [(conj result num) (+ s num)]
                (reduced [(conj result num) (+ s num)])))]
    (first (reduce add [[] 0] (sort > nums)))))
(map min-subsequence [[4 3 10 9 8] [4 4 7 6 7] [6]])
;;1408
(defn string-matching [words]
  (let [substrs (fn [i]
                  (let [needle (words i)]
                    (filter (fn [haystack]
                              (str/includes? haystack needle)) words)))
        is-substr (fn [i] (> (count (substrs i)) 1))]
    (->> (filter is-substr (range (count words)))
         (mapv #(get words %)))))
(map string-matching [["mass" "as" "hero" "superhero"] ["leetcode" "et" "code"] ["blue" "green" "bu"]])
;;1413
(defn min-start-value [nums]
  (let [sums (reduce (fn [results num]
                       (if (empty? results)
                         [num]
                         (conj results (+ (last results) num)))) [] nums)
        min-value (apply min sums)
        start-value (- 1 min-value)]
    (if (pos? start-value)
      start-value
      1)))
(map min-start-value [[-3 2 -3 4 2] [1 2] [1 -2 -3]])
;;1417
(defn reformat [s]
  (let [letters (vec (filter #(Character/isLetter %) (seq s)))
        digits (vec (filter #(Character/isDigit %) (seq s)))
        [short-list long-list] (if (< (count letters) (count digits)) [letters digits] [digits letters])
        abs (fn [n] (if (neg? n) (* -1 n) n))
        delta (abs (- (count letters) (count digits)))
        reformat' (fn [short-list long-list]
                    (let [result (reduce #(conj %1 (long-list %2) (short-list %2)) [] (range (count short-list)))]
                      (if (zero? delta)
                        result
                        (conj result (last long-list)))))]
    (str/join "" (if (<= delta 1)
                   (reformat' short-list long-list)
                   []))))
(map reformat ["a0b1c2" "leetcode" "1229857369" "covid2019" "ab123"])
;;1422
(defn max-score [s]
  (let [digits (vec s)
        count-score (fn [i]
                      (let [count-digit (fn [digit ds] (count (filter #(= % digit) ds)))
                            left (count-digit \0 (subvec digits 0 i))
                            right (count-digit \1 (subvec digits i (count digits)))]
                        (+ left right)))]
    (apply max (map count-score (range 1 (count digits))))))
(map max-score ["011101" "00111" "1111"])
;;1431
(defn kids-with-candies [candies extra-candies]
  (let [max-candies (apply max candies)]
    (map #(>= (+ % extra-candies) max-candies) candies)))
(map (partial apply kids-with-candies) ['([2 3 5 1 3] 3) '([4 2 1 1 2] 1) '([12 1 12] 10)])
;;1436
(defn dest-city [paths]
  (let [path-map (into {} paths)
        destination-city? (fn [city] (nil? (get path-map city)))]
    (first (filter destination-city? (vals path-map)))))
(map dest-city [[["London" "New York"] ["New York" "Lima"] ["Lima" "Sao Paulo"]]
                [["B" "C"] ["D" "B"] ["C" "A"]]
                [["A" "Z"]]])
;;1437
(defn k-length-apart [nums k]
  (let [k-length-apart? (fn [[result old] new]
                          (if (> (- new old) k)
                            [true new]
                            (reduced [false new])))
        one? (fn [[index num]] (= num 1))
        all-k-length-apart? (fn [indices] (reduce k-length-apart? [true (first indices)] (rest indices)))]
    (->> (map vector (range (count nums)) nums)
         (filter one?)
         (mapv first)
         (all-k-length-apart?)
         (first))))
(map (partial apply k-length-apart) ['([1 0 0 0 1 0 0 1] 2) '([1 0 0 1 0 1] 2) '([1 1 1 1 1] 0) '([0 1 0 1] 1)])
;;1441
(defn build-array [target n]
  (let [xs (range 1 (inc n))
        translate (fn [results x]
                    (if (= -1 (.indexOf target x))
                      (if (empty? results) results (concat ["Push" "Pop"] results))
                      (concat ["Push"] results)))]
    (reduce translate [] (reverse xs))))
(map (partial apply build-array) ['([1 3] 3) '([1 2 3] 3) '([1 2] 4) '([2 3 4] 4)])
;;1446
(defn max-power [s]
  (let [split-chars (fn [[results result] c]
                      (cond
                        (empty? result) [results [c]]
                        (= (last result) c) [results (conj result c)]
                        :else [(conj results result) [c]]))
        [results' result] (reduce split-chars [[] []] (vec s))
        results (if (empty? result)
                  results'
                  (conj results' result))]
    (apply max (map count results))))
(map max-power ["leetcode" "abbcccddddeeeeedcba" "triplepillooooow" "hooraaaaaaaaaaay" "tourist"])
;;1450
(defn busy-student [start-time end-time query-time]
  (let [busy? (fn [start end] (and (<= query-time start) (>= end query-time)))]
    (->> (map busy? start-time end-time)
         (filter true?)
         (count))))
(map (partial apply busy-student) ['([4] [4] 4) '([4] [4] 5) '([1 1 1 1] [1 3 2 4] 7) '([9 8 7 6 5 4 3 2 1] [10 10 10 10 10 10 10 10 10] 5)])
;;1455
(defn is-prefix-of-words [sentence search-word]
  (let [words (re-seq #"\w+" sentence)
        get-first-index (fn [indices] (if (pos? (count indices))
                                        (first indices)
                                        -1))
        prefix? (fn [[index word]] (str/starts-with? word search-word))]
    (->> (map vector (range 1 (inc (count words))) words)
         (filter prefix?)
         (map first)
         (get-first-index))))
(map (partial apply is-prefix-of-words) ['("i love eating burger" "burg")
                                         '("this problem is an easy problem" "pro")
                                         '("i am tired" "you")
                                         '("i use triple pillow" "pill")
                                         '("hello from the other side" "they")])
;;1460
(defn can-be-equal [target arr]
  (apply = (map sort [target arr])))
(map (partial apply can-be-equal) ['([1 2 3 4] [2 4 1 3]) '([7] [7]) '([3 7 9] [3 7 11]) '([1 1 1 1 1] [1 1 1 1 1])])
;;1464
(defn max-product [nums]
  (->> (take 2 (sort > nums))
       (map dec)
       (apply *)))
(map max-product [[3 4 5 2] [1 5 4 5] [3 7]])
;;1470
(defn shuffle1 [nums n]
  (flatten (map (fn [i] [(nums i) (nums (+ n i))]) (range n))))
(map (partial apply shuffle1) ['([2 5 1 3 4 7] 3) '([1 2 3 4 4 3 2 1] 4) '([1 1 2 2] 2)])
;;1475
(defn final-prices [prices]
  (let [get-final-price (fn [i]
                          (let [price (prices i)
                                get-discount (fn [_ discount]
                                               (if (<= discount price)
                                                 (reduced discount)
                                                 0))
                                discount (reduce get-discount 0 (subvec prices (inc i) (count prices)))]
                            (- price discount)))]
    (map get-final-price (range (count prices)))))
(map final-prices [[8 4 6 2 3] [1 2 3 4 5] [10 1 1 6]])
;;1480
(defn running-sum [nums]
  (reduce (fn [results num]
            (conj results (+ (last results) num)))
          [(first nums)] (rest nums)))
(map running-sum [[1 2 3 4] [1 1 1 1 1] [3 1 2 10 1]])
;;1486
(defn xor-operation [n start]
  (let [nums (map (fn [i] (+ start (* 2 i))) (range n))
        xor (fn [result num] (bit-xor result num))]
    (reduce xor (first nums) (rest nums))))
(map (partial apply xor-operation) ['(5 0) '(4 3) '(1 7) '(10 5)])
;;1491
(defn average [salary]
  (let [sum' (apply + salary)
        max-salary (apply max salary)
        min-salary (apply min salary)
        sum (- sum' max-salary min-salary)
        num (- (count salary) 2)]
    (float (/ sum num))))
(map average [[4000 3000 1000 2000] [1000 2000 3000] [6000 5000 4000 3000 2000 1000] [8000 9000 2000 3000 6000 1000]])
;;1496
(defn is-path-crossing [path]
  (let [commands (vec path)
        move (fn [[points [x y] crossing] command]
               (let [point
                     (case command
                       \N [x (inc y)]
                       \S [x (dec y)]
                       \E [(inc x) y]
                       \W [(dec x) y])]
                 (if (contains? points point)
                   (reduced [points point true])
                   [(conj points point) point false])))]
    (last  (reduce move [#{[0 0]} [0 0] false] commands))))
(map is-path-crossing ["NES" "NESWW"])
;;1502
(defn can-make-arithmetic-progression [arr]
  (let [xs (vec (sort arr))
        delta (- (second xs) (first xs))
        compare-difference (fn [[_ prev] curr]
                             (if (= delta (- curr prev))
                               [true delta curr]
                               (reduced [false delta curr])))]
    (first  (reduce compare-difference [true (second xs)] (drop 2 xs)))))
(map can-make-arithmetic-progression [[3 5 1] [1 2 4]])

;;1507
(defn reformat-date [date]
  (let [[d m y] (re-seq #"\w+" date)
        months {"Jan" "01" "Feb" "02"  "Mar" "03" "Apr" "04" "May" "05" "Jun" "06" "Jul" "07" "Aug" "08" "Sep" "09" "Oct" "10"  "Nov" "11" "Dec" "12"}
        month (get months m)
        day (if (= 3 (count d))
              (str "0" (subs d 0 1))
              (subs d 0 2))]
    (str y "-" month "-" day)))
(map reformat-date ["20th Oct 2052" "6th Jun 1933" "26th May 1960"])
;;1512
(defn num-identical-pairs [nums]
  (let [freqs (frequencies nums)]
    (->> (filter #(> % 1) (vals freqs))
         (map (fn [i] (quot (* i (dec i)) 2)))
         (apply +))))
(map num-identical-pairs [[1,2,3,1,1,3] [1,1,1,1] [1,2,3]])
;;1518
(defn num-water-bottles [num-bottles num-exchange]
  (letfn [(exchange [water-bottles empty-bottles]
            (let [bottles (+ water-bottles empty-bottles)
                  remaining #(exchange (quot bottles num-exchange) (rem bottles num-exchange))]
              (if (and (zero? water-bottles) (< bottles num-exchange))
                0
                (+ water-bottles (remaining)))))]
    (exchange num-bottles 0)))
(map (partial apply num-water-bottles) ['(9 3) '(15 4) '(5 5) '(2 3)])
;;1523
(defn count-odds [low high]
  (let [len (inc (- high low))]
    (cond
      (even? len) (quot len 2)
      (odd? low) (inc (quot len 2))
      :else (quot len 2))))
(map (partial apply count-odds) ['(3 7) '(8 10)])
;;1528
(defn restore-string [s indices]
  (let [cs (vec s)
        result (make-array Character/TYPE (count s))
        indices-chars (map vector indices cs)]
    (doseq [[index c] indices-chars]
      (aset result index c))
    (str/join "" result)))
(map (partial apply restore-string) ['("abc" [0 1 2]) '("aiohn" [3 1 4 2 0]) '("aaiougrt" [4 0 2 6 7 3 1 5]) '("art" [1 0 2])])
;;1534
(defn count-good-triplets [arr a b c]
  (let [abs (fn [n] (if (neg? n) (* -1 n) n))
        len (count arr)
        indices (for [i (range (- len 2)) j (range (inc i) (- len 1)) k (range (inc j) len)]
                  (vector i j k))
        triplets? (fn [[i j k]]
                    (let [v1 (arr i)
                          v2 (arr j)
                          v3 (arr k)]
                      (and (>= a (abs (- v1 v2))) (>= b (abs (- v2 v3))) (>= c (abs (- v3 v1))))))]
    (reduce (fn [result [i j k]]
              (if (triplets? [i j k])
                (inc result)
                result)) 0 indices)))
(map (partial apply count-good-triplets) ['([3,0,1,1,9,7] 7 2 3) '([1,1,2,2,3] 0 0 1)])
;;1539
(defn find-kth-positive [arr k]
  (let [s (set arr)]
    (last (take k (filter #(not (contains? s %)) (range 1 1001))))))
(map (partial apply find-kth-positive) ['([2 3 4 7 11] 5) '([1 2 3 4] 2)])
;;1544 ;;TODO it isn't the best.
(defn make-good [s]
  (let [letters "abcdefghijklmnopqrstuvwxyz"
        lowers (vec letters)
        uppers (vec (str/upper-case letters))
        patterns (map (partial str/join "") (map vector lowers uppers))]
    (letfn [(make-good' [s]
              (let [s1 (reduce (fn [result pattern] (str/replace result pattern "")) s patterns)]
                (if (= s s1)
                  s
                  (make-good' s1))))]
      (make-good' s))))
(map make-good ["leEeetcode" "abBAcC" "s"])
;;(str/lower-case \a)
(defn three-consective-odds [arr]
  (let [count-odds (fn [result num]
                     (if (odd? num)
                       (if (= result 2)
                         (reduced 3)
                         (inc result))
                       0))]
    (<= 3 (reduce count-odds 0 arr))))
(map three-consective-odds [[2 6 4 1] [1 2 34 3 4 5 7 23 12]])
;;1556
(defn thousand-separator [n]
  (let [ds (vec (reverse (str n)))
        add-separator (fn [result i]
                        (if (= 2 (rem i 3))
                          (if (= i (dec (count ds)))
                            (conj result (ds i))
                            (conj result (ds i) \.))
                          (conj result (ds i))))]

    (->> (reduce add-separator [] (range (count ds)))
         (reverse)
         (str/join ""))))
(map thousand-separator [987 1234 123456789 0])
;;1560
(defn most-visited [n rounds]
  (let [counter (make-array Integer/TYPE n)
        expand-sectors (fn [start end]
                         (if (> start end)
                           (concat (range (inc start) (inc n)) (range 1 (inc end)))
                           (range (inc start) (inc end))))
        sections (cons (first rounds) (flatten (map expand-sectors (drop-last rounds) (rest rounds))))
        freqs (frequencies sections)
        compare-freq (fn [l r]
                       (let [result (compare (last r) (last l))]
                         (if (zero? result)
                           (compare (first l) (first r))
                           result)))
        max-value (apply max (vals freqs))]
    (map first (filter #(= max-value (last %)) (sort compare-freq freqs)))))
;)
(map (partial apply most-visited) ['(4 [1 3 1 2]) '(2 [2 1 2 1 2 1 2 1 2]) '(7 [1 3 5 7])])
;;1566
(defn contains-pattern [arr m k]
  (let [get-single-pattern (fn [coll start len]
                             (subvec coll start (+ start len)))
        get-pattern (fn [p len] (flatten (take len (repeat p))))
        check-pattern (fn [_ i]
                        (let [s (get-single-pattern arr i m)
                              end (+ i (* m k))
                              p1 (get-pattern s k)
                              p2 (subvec arr i end)]
                          (if (= p1 p2)
                            (reduced true)
                            false)))]
    (reduce check-pattern false (range (- (count arr) (* m k))))))
(map (partial apply contains-pattern) ['([1 2 4 4 4 4] 1 3) '([1 2 1 2 1 1 1 3] 2 2) '([1 2 1 2 1 3] 2 3) '([1 2 3 1 2] 2 2) '([2 2 2 2] 2 3)])
;1572
(defn diagonal-sum [mat]
  (let [len (count mat)
        indices1 (map (fn [i] [i i]) (range len))
        indices2 (map (fn [i] [i (- (dec len) i)]) (range len))
        indices (set (concat indices1 indices2))]
    (apply + (map (fn [[r c]] ((mat r) c)) indices))))
(map diagonal-sum [[[1 2 3]
                    [4 5 6]
                    [7 8 9]]
                   [[1 1 1 1]
                    [1 1 1 1]
                    [1 1 1 1]
                    [1 1 1 1]]
                   [[5]]])
;;1576
(defn modify-string [s]
  (let [letters (set (vec "abcdefghijklmnopqrstuvwxyz"))
        cs (into-array (vec s))
        get-non-repeating-char (fn [i]
                                 (let [len (count cs)
                                       neighbours
                                       (cond
                                         (= 1 len) []
                                         (zero? i) [(get cs (inc i))]
                                         (= i (dec len)) [(aget cs (dec i))]
                                         :else [(aget cs (dec i)) (aget cs (inc i))])]
                                   (first (set/difference letters (set neighbours)))))]
    (reduce (fn [result index]
              (let [c (aget result index)
                    value (if (= c \?)
                            (get-non-repeating-char index)
                            c)]
                (aset result index value)
                result))
            cs
            (range (count cs)))
    (str/join "" (vec cs))))
(map modify-string ["?zs" "ubv?w" "j?qg??b" "??yw?ipkj?"])
;;1582
(defn num-special [mat]
  (let [rows (count mat)
        cols (count (mat 0))
        sum-of-row (fn [r] (apply + (mat r)))
        sum-of-col (fn [c] (apply + (map (fn [r] ((mat r) c)) (range rows))))
        indices (for [r (range rows) c (range cols)] [r c])
        special? (fn [[r c]] (= 1 ((mat r) c) (sum-of-row r) (sum-of-col c)))]
    (->> (filter special? indices)
         (count))))
(map num-special [[[1 0 0]
                   [0 0 1]
                   [1 0 0]]
                  [[1 0 0]
                   [0 1 0]
                   [0 0 1]]
                  [[0 0 0 1]
                   [1 0 0 0]
                   [0 1 1 0]
                   [0 0 0 0]]
                  [[0 0 0 0 0]
                   [1 0 0 0 0]
                   [0 1 0 0 0]
                   [0 0 1 0 0]
                   [0 0 0 1 1]]])
;;1588
(defn sum-odd-length-subarrays1 [arr]
  (let [subarray (fn [len] (map #(subvec arr % (+ % len)) (range (inc (- (count arr) len)))))
        sum-subarray (fn [len] (apply + (flatten (subarray len))))]
    (apply + (map sum-subarray (range 1 (inc (count arr)) 2)))))

(defn sum-odd-length-subarrays [arr]
  (let [n (count arr)
        get-freq (fn [i] (quot (inc (* (- n i) (inc i))) 2))
        sum (fn [i] (* (arr i) (get-freq i)))]
    (apply + (map sum (range n)))))
(map sum-odd-length-subarrays [[1 4 2 5 3] [1 2] [10 11 12]])

;;1592
(defn reorder-spaces [text]
  (let [words (re-seq #"\w+" text)
        spaces (count (filter #(= \  %) (vec text)))
        num-of-separators (dec (count words))
        separator (if (zero? num-of-separators) "" (str/join "" (take (quot spaces num-of-separators) (repeat " "))))
        num-of-extra-spaces (if (zero? num-of-separators) num-of-separators (rem spaces num-of-separators))
        extra (if (zero? num-of-extra-spaces) "" (str/join "" (take (rem spaces num-of-separators) (repeat " "))))]
    (str (str/join separator words) extra)))
(map reorder-spaces ["  this   is  a sentence " " practice   makes   perfect" "hello   world" "  walks  udp package   into  bar a"])
;;1598
(defn min-operations [logs]
  (let [cd (fn [components log]
             (cond
               (= "../" log) (if (empty? components) [] (drop-last components))
               (= "./" log) components
               :else (conj components (subs log 0 (dec (count log))))))]
    (count (reduce cd [] logs))))
(map min-operations [["d1/" "d2/" "../" "d21/" "./"] ["d1/" "d2/" "./" "d3/" "../" "d31/"] ["d1/" "../" "../" "../"]])
;;1608
(defn special-array1 [nums]
  (let [special? (fn [x] (= x (count (filter #(>= % x) nums))))
        specials (filter special? (range (inc (count nums))))]
    (or (first specials) -1)))

(defn special-array [nums]
  (let [xs (vec (sort > nums))
        count-special (fn [r i]
                        (if (> i r)
                          (inc r)
                          r))
        r (reduce count-special 0 xs)
        len (count xs)
        not-special (and (< r len) (= r (xs r)))]
    (if not-special
      -1
      r)))
(map special-array [[3 5] [0 0] [0 4 3 0 4] [3 6 7 7 0]])
;;1614
(defn max-depth [s]
  (let [cs (vec (filter #(or (= \( %) (= \) %)) (seq s)))
        max-depth' (fn [[max-value depth] c]
                     (if (= \( c) [(max max-value (inc depth)) (inc depth)]
                                  [max-value (dec depth)]))]
    (first (reduce max-depth' [0 0] cs))))
(map max-depth ["(1+(2*3)+((8)/4))+1" "(1)+((2))+(((3)))" "1+(2*3)/(2-1)" "1"])
;;1619
(defn trim-mean [arr]
  (let [xs (sort arr)
        len (count arr)
        num (quot len 20)
        average (fn [xs] (float (/ (apply + xs) (count xs))))
        remaining (drop num (take (- len num) xs))]
    (average remaining)))
(map trim-mean [[1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3] [6,2,7,5,1,2,0,3,10,2,5,0,5,5,0,8,7,6,8,0]
                [6,0,7,0,7,5,7,8,3,4,0,7,8,1,6,8,1,1,2,4,8,1,9,5,4,3,8,5,10,8,6,6,1,0,6,10,8,2,3,4]
                [9,7,8,7,7,8,4,4,6,8,8,7,6,8,8,9,2,6,0,0,1,10,8,6,3,3,5,1,10,9,0,7,10,0,10,4,1,10,6,9,3,6,0,0,2,7,0,6,7,2,9,7,7,3,0,1,6,1,10,3]
                [4,8,4,10,0,7,1,3,7,8,8,3,4,1,6,2,1,1,8,0,9,8,0,3,9,10,3,10,1,10,7,3,2,1,4,9,10,7,6,4,0,8,5,1,2,1,6,2,5,0,7,10,9,10,3,7,10,5,8,5,7,6,7,6,10,9,5,10,5,5,7,2,10,7,7,8,2,0,1,1]])
;;1624
(defn max-length-between-equal-character [s]
  (let [freqs (frequencies s)
        chars (map first (filter (fn [[k v]] (> v 1)) freqs))
        max-length-substring (fn [c] (subs s (inc (str/index-of s c)) (str/last-index-of s c)))
        lens (map count (map max-length-substring chars))]
    (if (empty? lens)
      -1
      (apply max lens))))
(map max-length-between-equal-character ["aa" "abca" "cbzxy" "cabbac"])
;;1629
(defn slowest-key [release-times key-pressed]
  (let [ks (vec key-pressed)
        add-key-duration (fn [results index]
                           (let [key (ks index)
                                 duration (if (zero? index)
                                            (release-times index)
                                            (- (release-times index) (release-times (dec index))))]
                             (conj results [key duration])))
        key-durations (reduce add-key-duration [] (range (count ks)))
        longest-duration (apply max (map last key-durations))]
    (->> (filter #(= (last %) longest-duration) key-durations)
         (map first)
         (sort (fn [l r] (compare (int r) (int l))))
         (first))))
(map (partial apply slowest-key) ['([9,29,49,50] "cbcd") '([12,23,36,46,62] "spuda")])
;;1636
(defn frequency-sort [nums]
  (let [freqs (frequencies nums)]
    (->> (map (fn [num] [num (freqs num)]) nums)
         (sort (fn [l r] (compare (last l) (last r))))
         (map first))))
(map frequency-sort [[1 1 2 2 2 3] [2 3 1 3 2] [-1 1 -6 4 5 -6 1 4 1]])
;;1640
(defn can-form-array [arr pieces]
  (let [piece-map (reduce #(assoc %1 (first %2) %2) {} pieces)]
    (letfn [(can-form-array' [nums]
              (if (empty? nums)
                true
                (let [num (first nums)
                      piece (get piece-map num)
                      len #(count piece)
                      not-found? (or (nil? piece) (< (count nums) (len)) (not= piece (subvec nums 0 (len))))]
                  (if not-found?
                    false
                    (can-form-array' (subvec nums (len) (count nums)))))))]
      (can-form-array' arr))))
(map (partial apply can-form-array) ['([85] [[85]]) '([15 88] [[88] [15]])
                                     '([49 18 16] [[16 18 49]])
                                     '([91 4 64 78] [[78] [4 64] [91]])
                                     '([1 3 5 7] [[2 4 6 8]])])
;;1646
(defn get-maximum-generated [n]
  (let [xs (make-array Integer/TYPE (inc n))]
    (letfn [(generate-array-element [i]
              (let [half (quot i 2)]
                (cond
                  (or (= i 0) (= i 1)) (do (aset xs i i) (aget xs i))
                  (odd? i) (aset xs i (+ (aget xs half) (aget xs (inc half))))
                  :else (aset xs i (aget xs half)))))]
      (apply max (map generate-array-element (range (inc n)))))))
(map get-maximum-generated [7 2 3])
;;1652
(defn decrypt1 [code k]
  (let [len (count code)
        decode-zero (fn [code k]
                      (take (count code) (repeat 0)))
        decode-positive (fn [code k]
                          (let [get-nth-element (fn [i]
                                                  (->>  (if (< len (+ i k 1))
                                                          (concat (subvec code (inc i) len) (subvec code 0 (rem (+ i k 1) len)))
                                                          (subvec code (inc i) (+ i k 1)))
                                                        (apply +)))]
                            (map get-nth-element (range len))))
        decode-negative (fn [code k]
                          (let [get-nth-element (fn [i]
                                                  (let [start (+ i k)]
                                                    (->> (if (neg? start)
                                                           (concat (subvec code (+ start len) len)
                                                                   (subvec code 0 i))
                                                           (subvec code start i))
                                                         (apply +))))]
                            (map get-nth-element (range len))))]
    (cond
      (pos? k) (decode-positive code k)
      (neg? k) (decode-negative code k)
      :else (decode-zero code 0))))

(defn decrypt [code k]
  (let [len (count code)
        xs (make-array Integer/TYPE len)
        generate-nth-element (fn [i]
                               (let [start (if (pos? k) (rem (inc i) len) (rem (+ i k len) len))
                                     end (if (pos? k) (+ start k) (- start k))
                                     add (fn [result index]
                                           (+ result
                                              (code (rem index len))))]
                                 (reduce add 0 (range start end))))
        decode (fn [code k]
                 (map generate-nth-element (range len)))]
    (if (zero? k)
      (vec xs)
      (decode code k))))
(map (partial apply decrypt) ['([5 7 1 4] 3) '([1 2 3 4] 0) '([2 4 9 3] -2)])

;;1662
(defn array-strings-are-equal1 [word1 word2]
  (apply = (map #(str/join "" %) [word1 word2])))

(defn array-strings-are-equal [word1 word2]
  (letfn [(not-equal-length? [word1 word2]
            (let [length (fn [ws]
                           (apply + (map count ws)))]
              (not= (length word1) (length word2))))
          (compare-string [word-index1 index1 word-index2 index2]
            (let [get-char (fn [words word-index index]
                             (get (words word-index) index))]
              (cond
                (and (= word-index1 (count word1)) (zero? index1) (= word-index2 (count word2)) (zero? index2)) true
                (not= (get-char word1 word-index1 index1) (get-char word2 word-index2 index2)) false
                :else (let [index1' (rem (inc index1) (count (word1 word-index1)))
                            index2' (rem (inc index2) (count (word2 word-index2)))
                            word-index1' (if (zero? index1') (inc word-index1) word-index1)
                            word-index2' (if (zero? index2') (inc word-index2) word-index2)]
                        (compare-string word-index1' index1' word-index2' index2')))))]
    (if (not-equal-length? word1 word2)
      false
      (compare-string 0 0 0 0))))
(map (partial apply array-strings-are-equal) ['(["ab" "c"] ["a" "bc"]) '(["a" "cb"] ["ab" "c"]) '(["abc" "d" "defg"] ["abcddefg"])])

;;1668
(defn max-repeating [sequence word]
  (letfn [(max-repeating' [haystack needle start]
            (let [index (.indexOf (subs haystack start) needle)]
              (if (or (= index -1) (>= start (count haystack)))
                0
                (inc (max-repeating' haystack needle (+ start (count needle)))))))]
    (max-repeating' sequence word 0)))
(map (partial apply max-repeating) ['("ababc" "ab") '("ababc" "ba") '("ababc" "ac")])
;;1672
(defn maximum-wealth [accounts]
  (apply max (map #(apply + %) accounts)))
(map maximum-wealth [[[1 2 3] [3 2 1]] [[1 5] [7 3] [3 5]] [[2 8 7] [7 1 3] [1 9 5]]])
;;1678
(defn interpret1 [command]
  (let [rules {"()" "o" "(al)" "al"}
        replace-str (fn [result src]
                      (str/replace result src (get rules src)))]
    (reduce replace-str command (keys rules))))
(map interpret1 ["G()(al)" "G()()()()(al)" "(al)G(al)()()G"])
;;1684
(defn count-consistent-strings [allowed words]
  (let [distinct-chars (fn [s] (set (vec s)))
        allowed-chars (distinct-chars allowed)
        consistent-string? (fn [word]
                             (let [chars (distinct-chars word)]
                               (set/subset? chars allowed-chars)))]
    (count (filter consistent-string? words))))
(map (partial apply count-consistent-strings) ['("ab" ["ad" "bd" "aaab" "baa" "badab"])
                                               '("abc" ["a" "b" "c" "ab" "ac" "bc" "abc"])
                                               '("cad" ["cc" "acd" "b" "ba" "bac" "bad" "ac" "d"])])
;;1688
(defn number-of-matches [n]
  (letfn [(number-of-matches' [n]
            (let [half (quot n 2)
                  extra (if (even? n) 0 1)]
              (if (= n 1)
                0
                (+ half extra (number-of-matches' half)))))]
    (number-of-matches' n)))
(map number-of-matches [7 14])
;;1694
(defn reformat-number [number]
  (let [digits (vec (re-seq #"[0-9]" number))
        len (count digits)
        slashes (let [remaider (rem len 3)
                      num (quot len 3)]
                  (cond
                    (= 1 remaider) (dec num)
                    (= 2 remaider) num
                    :else (dec num)))
        insert-dash (fn [result i]
                      (let [start (* i 3)
                            end (+ start 3)]
                        (concat result (subvec digits start end) ["-"])))
        part1 (reduce insert-dash [] (range slashes))
        part2 (let [ds (vec (drop (* slashes 3) digits))]
                (if (= 4 (count ds))
                  (concat (subvec ds 0 2) ["-"] (subvec ds 2 4))
                  ds))]
    (str/join "" (concat part1 part2))))
(map reformat-number ["1-23-45 6" "123 4-567" "123 4-5678"])
;;1700
(defn count-students [students sanwiches]
  (let [get-index  #(.indexOf students (first sanwiches))
        remove-student (fn [index]
                         (into (subvec students 0 index)
                               (subvec students (inc index))))]
    (cond
      (empty? sanwiches) 0
      (>= (get-index) 0) (count-students (remove-student (get-index)) (rest sanwiches))
      :else (count students))))
(map (partial apply count-students) ['([1 1 0 0] [0 1 0 1]) '([1 1 1 0 0 1] [1 0 0 0 1 1])])
;;1704
(defn halves-are-alike [s]
  (let [half-len (quot (count s) 2)
        count-vowels (fn [s]
                       (->> (str/lower-case s)
                            (vec)
                            (filter #(>= (.indexOf "aeiou" (str %)) 0))
                            (count)))
        get-nth-half #(subs s (* % half-len) (* (inc %) half-len))]
    (->> (map get-nth-half (range 2))
         (map count-vowels)
         (apply =))))
(map halves-are-alike ["book" "textbook" "MerryChristmas" "AbCdEfGh"])
;;2021-10-07
;;1710
(defn maximum-units [box-types truck-size]
  ;  (let [dp (make-array Integer/TYPE (count box-types) (count (box-types 0)))]
  (let [boxes (sort (fn [l r] (compare (last r) (last l))) box-types)]
    (letfn [(maximum' [boxes size]
              (if (or (empty? boxes) (zero? size))
                0
                (let [num (first (first boxes))
                      units (last (first boxes))]
                  (if (>= num size)
                    (* size units)
                    (+ (* num units) (maximum' (rest boxes) (- size num)))))))]
      (maximum' boxes truck-size))))
(map (partial apply maximum-units) ['([[1 3] [2 2] [3 1]] 4) '([[5 10] [2 5] [4 7] [3 9]] 10)])
;;1716
(defn total-money [n]
  (let [add (fn [results i]
              (if (< i 7) (conj results (inc i))
                          (conj results (inc (results (- i 7))))))]
    (apply + (reduce add [] (range n)))))
(map total-money [4 10 20])
;;1720
(defn decode [encoded fst]
  (let [decode' (fn [results num]
                  (conj results
                        (bit-xor num (last results))))]
    (reduce decode' [fst] encoded)))
(map (partial apply decode) ['([1 2 3] 1) '([6 2 7 3] 4)])
;;1725
(defn count-good-rectangles [rectangles]
  (let [squares (map #(apply min %) rectangles)
        maxlen (apply max squares)]
    (get (frequencies squares) maxlen)))
(map count-good-rectangles [[[5 8] [3 9] [5 12] [16 5]] [[2 3] [3 7] [4 3] [3 7]]])
;;1732
(defn largest-altitude [gain]
  (let [add-gain (fn [results g]
                   (conj results (+ g (last results))))]
    (apply max (reduce add-gain [0] gain))))
(map largest-altitude [[-5 1 5 0 -7] [-4 -3 -2 -1 4 3 2]])

;;1732
(defn maximum-time1 [time]
  (let [to-integer (fn [ds] (reduce #(+ (* %1 10) %2) 0 ds))
        get-hours (fn [ds] (to-integer (subvec ds 0 2)))
        get-minutes (fn [ds] (to-integer (subvec ds 2 4)))
        valid-time? (fn [ds] (let [hours (get-hours ds)
                                   minutes (get-minutes ds)]
                               (and (>= hours 0) (<= hours 23) (>= minutes 0) (<= minutes 59))))
        digits (mapv #(if (= % \?)
                        -1
                        (- (int %) (int \0)))
                     (str/replace time ":" ""))
        to-range (fn [digit] (if (neg? digit) (range 10) [digit]))
        all-posible-digits (for [d0 (to-range (digits 0)) d1 (to-range (digits 1)) d2 (to-range (digits 2)) d3 (to-range (digits 3))]
                             [d0 d1 d2 d3])]
    (let [ds (mapv str (first (sort #(compare (to-integer %2) (to-integer %1)) (filter valid-time? all-posible-digits))))]
      ds
      (str/join "" (concat (subvec ds 0 2) [":"] (subvec ds 2 4))))))
;;1736
(defn maximum-time [time]
  (let [digits (vec time)
        d0' (digits 0)
        d1' (digits 1)
        d2' (digits 3)
        d3' (digits 4)

        d0 (cond (not= d0' \?) d0'
                 (= d1' \?) \2
                 (< (int d1') 4) \2
                 :else \1)
        d1 (cond (not= d1' \?) d1'
                 (= d0' \2) \3
                 :else \9)
        d2 (if (not= d2' \?)
             d2'
             \5)
        d3 (if (not= d3' \?)
             d3'
             \9)]
    (str/join "" [d0 d1 \: d2 d3])))
(map maximum-time ["2?:?0" "0?:3?" "1?:22"])
;;1742
(defn count-balls [low-limit high-limit]
  (letfn [(integer-digits [n]
            (if (< n 10)
              [n]
              (conj (integer-digits (quot n 10)) (rem n 10))))
          (sum-digits [n] (apply + (integer-digits n)))]
    (->> (range low-limit (inc high-limit))
         (map sum-digits)
         (frequencies)
         (into [])
         (sort (fn [l r] (compare (last r) (last l))))
         (first)
         (last))))
(map (partial apply count-balls) ['(1 10) '(5 15) '(19 28)])
;;1748
(defn sum-of-unique [nums]
  (->> (frequencies nums)
       (filter (fn [[num freq]]
                 (= freq 1)))
       (map first)
       (apply +)))
(map sum-of-unique [[1 2 3 2] [1 1 1 1 1] [1 2 3 4 5]])
;;1752
(defn check1 [nums]
  (let [to-string (fn [xs] (str/join " " (map str xs)))
        haystack (to-string (into nums nums))
        needle (to-string (vec (sort nums)))
        found? (fn [result] (not= -1 result))]
    (found? (.indexOf haystack needle))))

(defn check [nums]
  (let [len (count nums)
        count-not-sorted-pairs (fn [result index]
                                 (let [n1 (nums index)
                                       n2 (nums (rem (inc index) len))]
                                   (if (> n1 n2)
                                     (inc result)
                                     result)))
        originally-sorted? (fn [result] (< result 2))
        num-of-not-sorted-pairs (reduce count-not-sorted-pairs 0 (range len))]
    (originally-sorted? num-of-not-sorted-pairs)))
(map check [[3 4 5 1 2] [2 1 3 4] [1 2 3] [1 1 1] [2 1]])
;;1758
(defn min-operations [s]
  (let [digits (vec s)
        len (count s)
        zeros (count (filter #(= (digits %) \0) (range 0 len 2)))
        ones (count (filter #(= (digits %) \1) (range 1 len 2)))]
    (min (- len zeros ones) (+ zeros ones))))
(map min-operations ["0100" "10" "1111"])
;;1763
(defn longest-nice-substring [s]
  (let [freqs (frequencies s)
        upper-case #(first (vec (str/upper-case %)))
        lower-case #(first (vec (str/lower-case %)))
        nice? (fn [s]
                (let [chars (map str (keys (frequencies s)))
                      lowercase-chars (distinct (map str/lower-case chars))]
                  (and (pos? (count chars)) (= (* (count lowercase-chars) 2) (count chars)))))
        bad-char? (fn [c] (or (nil? (get freqs (upper-case c)))
                              (nil? (get freqs (lower-case c)))))
        bad-chars (filter bad-char? (keys freqs))
        pattern (str "[" (str/join "" bad-chars) "]+")
        compare-by-size (fn [l r]
                          (compare (count r) (count l)))
        longest-nice-substrings (fn [s] (sort compare-by-size
                                              (map longest-nice-substring
                                                   (str/split s (re-pattern pattern)))))]

    (if (empty? bad-chars)
      (if (nice? s) s "")
      (or (first (longest-nice-substrings s)) ""))))
(map longest-nice-substring ["YazaAay" "Bb" "c" "dDzeE"])
;;1768
(defn merge-alternately [word1 word2]
  (let [cs1 (vec word1)
        cs2 (vec word2)
        len1 (count cs1)
        len2 (count cs2)
        len (max (count cs1) (count cs2))
        merge (fn [result index]
                (let [get-next (fn [cs len index]
                                 (if (< index len)
                                   [(cs index)]
                                   []))
                      r1 (get-next cs1 len1 index)
                      r2 (get-next cs2 len2 index)]
                  (concat result r1 r2)))]
    (str/join "" (reduce merge [] (range len)))))
(map (partial apply merge-alternately) ['("abc" "pqr") '("ab" "pqrs") '("abcd" "pq")])
;;1773
(defn count-matches [items rule-key rule-value]
  (let [rules {"type" 0 "color" 1 "name" 2}
        find-item (fn [item]
                    (= rule-value (item (rules rule-key))))]
    (count (filter find-item items))))
(map (partial apply count-matches) ['([["phone" "blue" "pixel"] ["computer" "silver" "lenovo"] ["phone" "gold" "iphone"]]
                                      "color"
                                      "silver")
                                    '([["phone" "blue" "pixel"] ["computer" "silver" "phone"] ["phone" "gold" "iphone"]]
                                      "type"
                                      "phone")])
;;1779
(defn nearest-valid-point [x y points]
  (let [abs (fn [n] (if (neg? n)
                      (- 0 n)
                      n))
        distance (fn [[x1 y1] [x2 y2]]
                   (+ (abs (- x1 x2))
                      (abs (- y1 y2))))
        valid? (fn [[x1 y1]]
                 (or (= x1 x) (= y1 y)))
        compare-by-distance-then-index (fn [[index1 distance1] [index2 distance2]]
                                         (if (not= distance1 distance2)
                                           (compare distance1 distance2)
                                           (compare index1 index2)))]
    (->> (map vector (range (count points)) points)
         (filter (fn [[index point]]
                   (valid? point)))
         (map (fn [[index point]]
                [index (distance point [x y])]))
         (sort compare-by-distance-then-index)
         (#(or (first (first %)) -1)))))
(map (partial apply nearest-valid-point)
     ['(3 4 [[1 2] [3 1] [2 4] [2 3] [4 4]])
      '(3 4 [[3 4]])
      '(3 4 [[2 3]])])
;;1784
(defn check-ones-segment [s]
  (->> (re-seq #"1+" s)
       (map count)
       (filter #(> % 1))
       ((comp not empty?))))
(map check-ones-segment ["1001" "110"])
;;1790
(defn are-almost-equal [s1 s2]
  (->> (map = (vec s1) (vec s2))
       (filter false?)
       (count)
       (#(<= % 2))))
(map (partial apply are-almost-equal) ['("bank" "kanb") '("attack" "defend") '("kelb" "kelb") '("abcd" "dcba")])
;;1791
(defn find-center [edges]
  (let [compare-freq (fn [[num1 freq1] [num2 freq2]]
                       (compare freq2 freq1))]
    (->> (frequencies (flatten edges))
         (sort compare-freq)
         ((comp first first)))))
(map find-center [[[1 2] [2 3] [4 2]] [[1 2] [5 1] [1 3] [1 4]]])
;;1796
(defn second-highest [s]
  (let [second-largest (fn [digits] (if (> (count digits) 1)
                                      (second digits)
                                      -1))
        to-int (fn [c]
                 (Integer/parseInt (str c)))
        distinct-digits (fn [xs]
                          (let [ds (set (vec (str/join "" xs)))]
                            (map to-int ds)))]

    (->>  (re-seq #"[0-9]+" s)
          (distinct-digits)
          (sort >)
          (second-largest))))
(map second-highest ["dfa12321afd" "abc1111"])
;;1800
(defn max-ascending-sum [nums]
  (let [len (count nums)
        max-sum-subarray (fn [[max-sum sum] index]
                           (let [num (if (< index len) (nums index) 0)]
                             (cond
                               (= index len) [(max max-sum sum) 0]
                               (< (nums (dec index)) num) [max-sum (+ sum num)]
                               :else [(max max-sum sum) num])))
        initial-value [(first nums) (first nums)]]
    (first (reduce max-sum-subarray initial-value (range 1 (inc len))))))
(map max-ascending-sum [[10 20 30 5 10 50] [10 20 30 40 50] [12 17 15 13 10 11 12] [100 10 1]])

;; (defn twenty-four []
;;   (let [card-range (range 1 11)
;;         operators [+ - * /]
;;         cards-list (set (map sort (for [c0 card-range c1 card-range c2 card-range c3 card-range] [c0 c1 c2 c3])))
;;         operators-list (for [op0 operators op1 operators op2 operators] [op0 op1 op2])]
;;     (letfn [(permutations [xs]
;;               (if (empty? xs)
;;                 [[]]
;;                 (reduce (fn [[results result] x]
;;                           [] (concat results (map #(concat result [x] %) permutations (rest xs)))
;;                           ))
;;                         [] xs)))]

;;       (count cards-list)
;;       (count operators-list)
;; ;    (op0 c0 (op1 c1 (op2 c2 c3)))
;;       (permutations [1 2 3]))))
;; (twenty-four)
;; ;(map #(cons 1 %) [[]])
;;2021-10-08

(defn num-different-integers [word]
  (->> (re-seq #"\d+" word)
       (map #(Integer/parseInt %))
       (distinct)
       (count)))
(map num-different-integers ["a123bc34d8ef34" "leet1234code234" "a1b01c001"])
;;1812
(defn square-is-white [coordinates]
  (let [column-index-map {\a 1 \b 2 \c 3 \d 4 \e 5 \f 6 \g 7 \h 8}
        get-row (fn [c] (- (int c) (int \0)))
        get-col (fn [c] (get column-index-map c))
        cs (vec coordinates)
        row (get-row (last cs))
        col (get-col (first cs))]
    (odd? (+ row col))))
(map square-is-white ["a1" "h3" "c7"])
;;1816
(defn truncate-sentence [s k]
  (str/join " " (take k (re-seq #"\w+" s))))
(map (partial apply truncate-sentence) ['("Hello how are you Contestant" 4) '("What is the solution to this problem" 4) '("chopper is not a tanuki" 5)])
;;1822
(defn array-sign [nums]
  (let [sign (fn [n]
               (cond
                 (pos? n) 1
                 (neg? n) -1
                 :else 0))]

    (sign (reduce * 1 nums))))
(map array-sign [[-1,-2,-3,-4,3,2,1] [1,5,0,2,-3] [-1,1,-1,1,-1]])
;;1827
(defn min-operations [nums]
  (let [increase (fn [[result prev] num]
                   (if (> num prev)
                     [result num]
                     [(+ result (- (inc prev) num)) (inc prev)]))]
    (first (reduce increase [0 (first nums)] (rest nums)))))
(map min-operations [[1,1,1] [1,5,2,4,1] [8]])
;;1832
(defn check-if-pangram [sentence]
  (->> (seq sentence)
       (set)
       (count)
       (#(= 26 %))))
(map check-if-pangram ["thequickbrownfoxjumpsoverthelazydog" "leetcode"])
;;1837
(defn sum-base [n k]
  (letfn [(integer-digits [n k]
            (if (< n k)
              [n]
              (conj (integer-digits (quot n k) k) (rem n k))))]
    (apply + (integer-digits n k))))
(map (partial apply sum-base) ['(34 6) '(10 10)])

;;1844
(defn replace-digits [s]
  (let [cs (vec s)
        shift (fn [c x]
                (char (+ (int c) (- (int x) (int \0)))))]
    (letfn [(replace-digits' [cs]
              (cond
                (empty? cs) []
                (= 1 (count cs)) [(first cs)]
                (Character/isDigit (second cs)) (concat [(first cs) (shift (first cs) (second cs))]
                                                        (replace-digits' (drop 2 cs)))
                :else (cons (first cs)
                            (replace-digits' (rest cs)))))]
      (str/join "" (replace-digits' cs)))))
(map replace-digits ["a1c1e1" "a1b2c3d4e"])
;;1848
(defn get-min-distance [nums target start]
  (let [abs (fn [n] (if (neg? n) (- 0 n) n))]
    (->> (map vector (range (count nums)) nums)
         (filter (fn [[index num]] (= num target)))
         (map (fn [[index num]] [index (abs (- index start))]))
         (sort (fn [l r] (compare (last l) (last r))))
         (first)
         (last))))
(map (partial apply get-min-distance)
     ['([1,2,3,4,5] 5 3)
      '([1] 1 0)
      '([1,1,1,1,1,1,1,1,1,1] 1 0)])
;;1854
(defn maximum-population [logs]
  (let [compare-population-then-year (fn [[year1 num1] [year2 num2]]
                                       (if (= num1 num2)
                                         (compare year1 year2)
                                         (compare num2 num1)))]
    (->> (map #(apply range %) logs)
         (flatten)
         (frequencies)
         (into [])
         (sort compare-population-then-year)
         (ffirst))))
(map maximum-population [[[1993 1999] [2000 2010]] [[1950 1961] [1960 1971] [1970 1981]]])

;;1859
(defn sort-sentence [s]
  (->> (re-seq #"\w+" s)
       (map (fn [word]
              (let [len (count word)]
                [(subs word (dec len) len) (subs word 0 (dec len))])))
       (sort (fn [l r] (compare (first l) (first r))))
       (map last)
       (str/join " ")))
(map sort-sentence ["is2 sentence4 This1 a3" "Myself2 Me1 I4 and3"])
)
(defn -main []
  (defn animal-crossing []
    (let [animals (vec (sort [60 50 40]))
          max-load 90]
      (letfn [(can-travel? [animals]
                             (let [cnt (count animals)]
                               (or (= cnt 1) (<= (apply + animals) max-load))))
              (travel [result island-animals mainland-animals]
                (cond
                  (empty? island-animals) result
                  (> (count island-animals) 2) (map (fn [xs]
                                                      (let [island-animals' (vec (set/difference (set island-animals) (set xs)))
                                                            mainland-animals' xs]
                                                        (map #(travel result (conj island-animals' %) (vec (set/difference (set mainland-animals') (set [%])))) mainland-animals')))
                                                    (filter can-travel? (comb/combinations island-animals 2)))
                  :else (travel (conj result island-animals) [] mainland-animals)))]
      (travel [] animals [])
        )))
  ;(animal-crossing)
)
;(-main)
;(-main)
;; (defn billion-ai []
;;   (let [ai (fn [_ x] (if (= x "bye") (reduced nil) (println (str/replace x #"[吗|?|？]$" ""))))
;;         lines #(line-seq (java.io.BufferedReader. *in*))]
;;     (reduce ai nil (lines))))

;; (billion-ai)

;;1863
(defn subset-xor-sum [nums]
  (letfn [(xor [xs]
            (reduce bit-xor (first xs) (rest xs)))
          (permutations [xs]
            (let [len (count xs)
                  indices (for [i (range 0 (dec len)) j (range (inc i) len)]
                            [i j])]
              (letfn [(permutations' [i j]
                        (if (= len 1)
                          xs
                          (let [part1 (subvec xs i j)
                                part2 (subvec xs j len)]
                            (concat (permutations part1) (permutations part2)))))]
                (reduce (fn [results [i j]]
                          (concat results (permutations' i j))) [xs] indices))))]
    ;; (if (empty? nums)
    ;;   0
    ;;   (map xor (permutations nums)))
    (conj (permutations nums) [])))
(map subset-xor-sum [[1 3] [5 1 6] [3 4 5 6 7 8]])
;;1869
(defn check-zero-ones [s]
  (let [zeros (re-seq #"0+" s)
        ones (re-seq #"1+" s)
        max-length #(apply max (sort (map count %)))
        ]
    (> (max-length ones) (max-length zeros))
    )
)
(map check-zero-ones ["1101" "111000" "110100010"])

;;1876
(defn count-good-substring [s]
  (let [cs (vec s)
        good? (fn [i] (= 3 (count (set (subvec cs i (+ i 3))))))
        count-good-string (fn [result index] (if (good? index) (inc result) result))
        len (- (count s) 2)
        ]
    (reduce count-good-string 0 (range len))
    ))
(map count-good-substring ["xyzzaz" "aababcabc"])
(defn is-sum-equal [first-word second-word target-word]
  (let [letter-value (fn [c] (- (int c) (int \a)))
        numerical-value (fn [s] (reduce #(+ (* %1 10) %2) 0 (map letter-value (vec s))))
        [num1 num2 target] (map numerical-value [first-word second-word target-word])]
    (= (+ num1 num2) target)))
(map (partial apply is-sum-equal) ['("acb" "cba" "cdb") '("aaa" "a" "aab") '("aaa" "a" "aaaa")])
;;1886
(defn find-rotation [mat target]
  (let [rows (count mat)
        cols (count mat)
        n rows
        indices (for [r (range rows) c (range cols)] [r c])
        rotate (fn [[r c] index]
                 (case index
                   0 ((mat r) c)
                   1 ((mat c) (- n 1 r))
                   2 ((mat (- n 1 r)) (- n 1 c))
                   ((mat c) (- n 1 r))
                 ))
        compare-element (fn [index result [r c]]
                          (let [mat-element (rotate [r c] index)
                                target-element ((target r) c)]
                            (and result
                                 (= mat-element target-element))))
        compare-matrix (fn [result index]
                           (let [result (reduce #(compare-element index %1 %2) true indices)]
                             (if result
                               (reduced true)
                               false)
                           ))
        ]
    (reduce compare-matrix true (range 4))))
(map (partial apply find-rotation) ['([[0 1] [1 0]] [[1 0] [0 1]]) '([[0 1] [1 1]] [[1 0] [0 1]]) '([[0 0 0] [0 1 0] [1 1 1]] [[1 1 1] [0 1 0] [0 0 0]])])
;;1
(defn two-sum [nums target]
  (let [index-num-list (map vector (range (count nums)) nums)
        find-indices (fn [[result index-map] [index num]]
                       (let [index' (get index-map num)]
                         (if (nil? index')
                           [result (assoc index-map (- target num) index)]
                           (reduced [index' index]))))]
    (reduce find-indices [[] {}] index-num-list)))
(map (partial apply two-sum) ['([2,7,11,15] 9) '([3,2,4] 6) '([3 3] 6)])

(defn is-covered [ranges left right]
  (let [covered? (fn [[start end]]
                   (let [intervals (set (range start (inc end)))]
                     (or (contains? intervals left) (contains? intervals right))))
        check (fn [result [start end]]
                (if (covered? [start end])
                  (reduced true)
                  false))]
    (reduce check false ranges)))
(map (partial apply is-covered) ['([[1 2] [3 4] [5 6]] 2 5) '([[1 10] [10 20]] 21 21)])

(defn make-equal [words]
  (let [len (count words)]
    (->> (str/join "" words)
         (seq)
         (frequencies)
         (vals)
         (every? (fn [n] (zero? (rem n len)))))))
(map make-equal [["abc" "aabc" "bc"] ["ab" "a"]])

(defn largest-odd-number [num]
  ()
  (let [odd-char? (fn [[index c]] (odd? (- (int c) (int \0))))
        indices (->> (map vector (range (count num)) (seq num))
                     (filter odd-char?)
                     (map first))]
   (if (empty? indices)
     ""
     (subs num 0 (inc (last indices)))
)))
(map largest-odd-number ["52" "4206" "35427"])

;;1909
(defn can-be-increasing1 [nums]
  (let [more-than-one-duplicate (fn [nums]
                                  (> (- (count nums) (count (set nums))) 1))
        spike? (fn [x0 x1 x2]
                 (and (< x0 x2)
                      (or (>= x0 x1) (>= x1 x2))))
        count-spike (fn [result i]
                      (let [x0 (nums (dec i))
                            x1 (nums i)
                            x2 (nums (inc i))]
                        (if (spike? x0 x1 x2)
                          (inc result)
                          result)))
        spikes (reduce count-spike 0 (range 1 (- (count nums) 1)))

        rest-or-most-sorted? (fn [nums]
                               (or (= (rest nums) (sort < (rest nums)))
                                   (= (drop-last nums) (sort < (drop-last nums)))))]

    (cond
      (more-than-one-duplicate nums) false ;; false [1 1 1]
      (= 1 spikes) true                    ;; true  [1 2 10 5 7]
      :else (rest-or-most-sorted? nums)))) ;; true  [1 1] [962 23 27 555] [100 21 100] [262 138 583]
                                           ;; false [2 3 1 2]


(defn can-be-increasing2 [nums]
  (let [xs (into [0] nums)
        get-prev (fn [xs i prev]
                   (if (> (xs i) (xs (- i 2)))
                     (xs i)
                     prev))
        count-non-increasing (fn [[non-increasings prev] i]
                               (if (>= prev (xs i))
                                 [(inc non-increasings) (get-prev xs i prev)]
                                 [non-increasings prev]))
        non-increasings (first (reduce count-non-increasing [0 (first nums)] (range 2 (count xs))))]

     (< non-increasings 2)))


(defn can-be-increasing [nums]
  (let [increasing? (fn [index]
                      (let [xs (concat (subvec nums 0 index) (subvec nums (inc index)))]
                        (apply < xs)))]
    (reduce (fn [result index] (if (increasing? index)
                                 (reduced true)
                                 false)) false (range (count nums)))))

;; (defn can-be-increasing [nums]
;;   (let [nums (conj nums 1001)
;;         increasing? (fn [index]
;;                       (let [xs (concat (subvec nums 0 index) (subvec nums (inc index)))]
;;                         (apply < xs)))]
;;     (reduce (fn [result index] (if (>= (nums (dec index)) (nums index))
;;                                  (reduced (and (or (< (nums index) (nums (inc index))) (< (nums (dec index)) (nums (inc index)))) (apply < (subvec nums (inc index)))))
;;                                  true)) true (range 1 (count nums)))))
;;                        true           false   false  true    true       true           true             true       true         true
(map can-be-increasing [[1 2 10 5 7] [2 3 1 2] [1 1 1] [1 2 3] [1 1] [962 23 27 555] [449 354 508 962] [100 21 100] [262 138 583] [1 2 5 10 7]])
;;1913
(defn max-product-difference [nums]
  (let [xs (vec (sort < nums))
        len (count nums)
        max-value (apply * (subvec xs (- len 2) len))
        min-value (apply * (take 2 xs))]
    (- max-value min-value)))
(map max-product-difference [[5 6 2 7 4] [4 2 5 9 7 4 8]])

;;1920
(defn build-arrays [nums]
  (map #(nums (nums %)) (range (count nums))))
(map build-arrays [[0 2 1 5 3 4] [5 0 1 2 3 4]])

;;1925
(defn count-triples [n]
  (let [square-triples? (fn [[a b c]]
                          (= (+ (* a a) (* b b)) (* c c)))
        triples-list (for [a (range 1 (- n 1)) b (range (inc a) n) c (range (inc b) (inc n))]
                       [a b c])]
    (->> (filter square-triples? triples-list)
         (count)
         (#(* % 2)))))
(map count-triples [5 10])
;;1929
(defn get-concatenation [nums]
  (let [len (count nums)
        result (make-array Long/TYPE (* len 2))]
    (doseq [i (range (* len 2))]
      (aset result i (nums (rem i len))))
    (into [] result)))
(map get-concatenation [[1 2 1] [1 3 2 1]])

;;1935
(defn can-be-typed-words [text broken-letters]
  (let [broken-set (set (seq broken-letters))
        typeable? (fn [word]
                    (empty? (set/intersection
                             (set (seq word))
                             broken-set)))]
    (->> (re-seq #"\w+" text)
         (filter typeable?)
         (count))))
(map (partial apply can-be-typed-words) ['("hello world" "ad") '("leet code" "lt") '("leet code" "e")])

;;1941
(defn are-occurrence-equal [s]
  (->> (frequencies (seq s))
       (vals)
       (set)
       (#(= 1 (count %)))))
(map are-occurrence-equal ["abacbc" "aaabb"])

;;1945
(defn get-lucky [s k]
  (let [transform (fn [s _]
                    (->> (map (fn [d] (- (int d) (int \0))) (seq s))
                         (apply +)
                         (str)))]
    (->> (map (fn [c] (inc (- (int c) (int \a)))) (seq s))
         (map str)
         (str/join "")
         (#(reduce transform % (range k)))
         (Integer/parseInt))))
(map (partial apply get-lucky) ['("iiii" 1) '("leetcode" 2) '("zbax" 2)])
;;1952
(defn is-three [n]
  (let [divisable? (fn [a b]
                     (zero? (rem a b)))
        count-factors (fn [n] (reduce (fn [result num]
                                        (if (divisable? n num)
                                          (if (= num (quot n num))
                                            (inc result)
                                            (reduced (+ result 2)))
                                          result)) 0 (range 2 n)))]
    (if (< n 3)
      false
      (= 1 (count-factors n)))))
(map is-three [2 4])
;;1957
(defn make-fancy-string [s]
  (let [cs (vec s)
        transform (fn [[result prev cnt] c]
                    (cond
                      (not= prev c) [(conj result c) c 1]
                      (< cnt 2) [(conj result c) c (inc cnt)]
                      :else [result c cnt]))]
    (->> (reduce transform [[(first cs)] (first cs) 1] (rest cs))
         (first)
         (str/join ""))))
(map make-fancy-string ["leeetcode" "aaabaaaa" "aab"])

;;1961
(defn is-prefix-string [s words]
  (let [check-prefix-string (fn [[result prefix] word]
                              (let [new-prefix (str prefix word)]
                                (cond
                                  (= new-prefix s) (reduced true)
                                  (str/includes? s new-prefix) [false new-prefix]
                                  :else (reduced false))))]
    (reduce check-prefix-string [false ""] words)))
(map (partial apply is-prefix-string) ['("iloveleetcode" ["i" "love" "leetcode" "apples"])
                                       '("iloveleetcode" ["apples" "i" "love" "leetcode"])])
;;1967
(defn num-of-strings [patterns word]
  (let [substring? (fn [pattern] (>= (.indexOf word pattern) 0))]
  (count (filter substring? patterns))))
(map (partial apply num-of-strings) ['(["a" "abc" "bc" "d"] "abc") '(["a" "b" "c"] "aaaaabbbbb") '(["a" "a" "a"] "ab")])

;;1971
(defn valid-path [n edges start end]
  (let [add (fn [m [u v]]
              (assoc m u (conj (or (get m u) []) v)))
        all-edges (concat edges (mapv reverse edges))
        path-map (reduce add {} all-edges)]
    (letfn [(search [paths start end]
              (let [vertices (get path-map start)
                    search' (fn [result vertice]
                              (cond
                                (= vertice end) (reduced true)
                                (contains? paths vertice) false
                                :else (search (conj paths vertice) vertice end)))]
                (reduce search' false vertices)))]
      (search [start] start end))))
(map (partial apply valid-path) ['(3 [[0 1] [1 2] [2 0]] 0 2)
                                 '(6 [[0 1] [0 2] [3 5] [5 4] [4 3]] 0 5)])
;; ;;1974
(defn min-time-to-type [word]
  (let [abs (fn [n] (if (neg? n) (- 0 n) n))
        count-steps  (fn [[result prev] c]
                       (let [start (int prev)
                             end (int c)
                             delta (abs (- end start))
                             steps (min (- 26 delta) delta)]
                         [(+ result steps 1) c]))]
    (reduce count-steps [0 \a] (vec word))))
(map min-time-to-type ["abc" "bza" "zjpc"])

;;1979
(defn find-gcd [nums]
  (letfn [(gcd [a b]
            (if (zero? (rem a b))
              b
              (gcd b (rem a b))))]
    (gcd (apply max nums) (apply min nums))))
(map find-gcd [[2 5 6 9 10] [7 5 6 8 3] [3 3]])

;;1984
(defn minimum-difference [nums k]
  (let [xs (vec (sort nums))
        len (- (inc (count xs)) k)
        nums-list (reduce (fn [results i]
                        (conj results (subvec xs i (+ i k)))) [] (range len))]
      (apply min (map (fn [xs] (- (last xs) (first xs))) nums-list))
    ))
(map (partial apply minimum-difference) ['([90] 1) '([9 4 1 7] 2)])
;;1991
(defn find-middle-index [nums]
  (let [sum (apply + nums)
        find-index (fn [[result s] index]
                     (if (= (* 2 s) (- sum (nums index))) [index s]
                       [result (+ s (nums index))]))]
(first (reduce find-index [-1 0] (range 0 (count nums))))))
(map find-middle-index [[2 3 -1 8 4] [1 -1 4] [2 5] [1]])
;;1995
(defn count-quadruplets [nums]
  (let [len (count nums)
        indices (for [a (range (- len 3))
                      b (range (inc a) (- len 2))
                      c (range (inc b) (- len 1))
                      d (range (inc c) len)]
                  [a b c d])
        special-quadruplet? (fn [[a b c d]]
                                 (= (+ (nums a) (nums b) (nums c)) (nums d)))
        ]
   (count (filter special-quadruplet? indices))))

(map count-quadruplets [[1 2 3 6] [3 3 6 4 5] [1 1 1 3 5]])

;;2000
(defn reverse-prefix [word ch]
  (let [index (.indexOf word ch)
        reverse-prefix' (fn [word index] (str/join "" (reverse (subs word 0 (inc index)))))]
    (if (= -1 index)
      word
      (str (reverse-prefix' word index) (subs word (inc index))))))
(map (partial apply reverse-prefix) ['("abcdefd" "d") '("xyxzxe" "z") '("abcd" "z")])

;;2006
(defn count-k-difference [nums k]
  (let [len (count nums)
        abs (fn [n] (if (neg? n) (- 0 n) n))
        pairs (for [i (range (dec len)) j (range (inc i) len)]
                  [(nums i) (nums j)])
        k-difference? (fn [pair] (= k (abs (apply - pair))))]
    (count (filter k-difference? pairs))))
(map (partial apply count-k-difference) ['([1 2 2 1] 1) '([1 3] 3) '([3 2 1 5 4] 2)])

;;2011
(defn final-value-after-operations [operations]
  (->> (map #(if (>= (.indexOf % "+") 0) 1 -1) operations)
       (apply +)))
(map final-value-after-operations [["--X" "X++" "X++"] ["++X" "++X" "X++"] ["X++" "++X" "--X" "X--"]])

;;2016
(defn maximum-difference [nums]
  (let [max-difference (fn [[difference min-value] num]
                         [(max difference (- num min-value)) (min num min-value)])]
   (first (reduce max-difference [-1 (first nums)] (rest nums)))))
(map maximum-difference [[7 1 5 4] [9 4 3 2] [1 5 2 10]])

;;2022
(defn construct-2d-array [original m n]
  (let [construct-2d-array' (fn [orignal m n]
                              (let [matrix (make-array Long/TYPE m n)
                                    indices (for [r (range m) c (range n)] [r c])]
                                (doseq [[r c] indices]
                                  (aset matrix r c (orignal (+ (* r n) c))))
                                (mapv vec matrix)
                                )
                              )]
    (if (not= (count original) (* m n))
      []
      (construct-2d-array' original m n))))
(map (partial apply construct-2d-array) ['([1 2 3 4] 2 2) '([1 2 3] 1 3) '([1 2] 1 1) '([3] 1 2)])

;;2027
(defn minimum-moves [s]
  (let [xs (re-seq #"X+" s)
        get-moves #(int (Math/ceil (/ % 3)))
        get-all-moves (fn [xs]
                    (->> (map count xs)
                         (map get-moves)
                         (apply +)))]
    (if (empty? xs)
      0
      (get-all-moves xs))))
(map minimum-moves ["XXX" "XXOX" "OOOO"])

;;2032
(defn two-out-of-three [nums1 nums2 nums3]
  (let [intersection (fn [xs ys]
                       (vec (set/intersection (set xs) (set ys))))
        array-list [(intersection nums1 nums2) (intersection nums2 nums3) (intersection nums1 nums3)]]
    (vec (set (flatten array-list)))))
(map (partial apply two-out-of-three) ['([1 1 3 2] [2 3] [3]) '([3 1] [2 3] [1 2]) '([1 2 2] [4 3 3] [5])])

;;157
(defn read-n-chars [file n]
  (let [len (count file)]
    (if (>= n len)
      len
      n)))
(map (partial apply read-n-chars) ['("abc" 4) '("abcde" 5) '("abcdABCD1234" 12) '("leetcode" 5)])

;;163
(defn find-missing-ranges [nums lower upper]
  (let [xs (vec (sort (distinct (conj (cons (dec lower) nums) (inc upper)))))
        find-missing (fn [results index]
                       (let [start (inc (xs (dec index)))
                             end (dec (xs index))]
                         (cond
                           (= start end) (conj results (str start))
                           (< start end) (conj results (str start "->" end))
                           :else results)))]
      (reduce find-missing [] (range 1 (count xs)))))
(map (partial apply find-missing-ranges) ['([0 1 3 50 75] 0 99) '([] 1 1) '([] -3 -1) '([-1] -1 -1) '([-1] -2 -1)])

;;243
(defn shortest-distance [words-dict word1 word2]
  (let [word-indices (mapv vector (range (count words-dict)) words-dict)
        word1-indices (map first (filter (fn [[index word]] (= word word1)) word-indices))
        word2-indices (map first (filter (fn [[index word]] (= word word2)) word-indices))
        indices (for [i word1-indices j word2-indices] [i j])
        abs (fn [n] (if (neg? n) (- 0 n) n))
        minimum-distance (fn [distance [i j]] (min distance (abs (- i j))))]
    (reduce minimum-distance (count words-dict) indices)))
(map (partial apply shortest-distance) ['(["practice"  "makes"  "perfect"  "coding"  "makes"] "coding" "practice")
     '(["practice"  "makes"  "perfect"  "coding"  "makes"] "makes" "coding")])

;;246
(defn is-strobogrammatic [num]
  (let [len (count num)
        digits (vec num)
        digit-map {\6 \9 \9 \6 \8 \8 \1 \1}
        check (fn [result i]
                (let [left (digits i)
                      digit (get digit-map left)
                      right (digits (- len 1 i))]
                  (if (or (nil? digit) (not= digit right))
                    (reduced false)
                    result)))]
    (reduce check true (range (quot len 2)))))
(map is-strobogrammatic ["69" "88" "962" "1"])

;;252
(defn can-attend-meetings [intervals]
  (let [interval-list (vec (sort (fn [l r]
                              (compare (first l) (first r))) intervals))
        attend? (fn [result index]
                  (let [end (last (interval-list (dec index)))
                        start (first (interval-list index))]
                  (if (> end start)
                    (reduced false)
                    result)))]

    (reduce attend? true (range 1 (count interval-list)))))
(map can-attend-meetings [[[0 30] [5 10] [15 20]] [[7 10] [2 4]]])

;;262
(defn can-permute-palindrome [s]
  (let [lens (vals (frequencies (seq s)))
        odds (count (filter odd? lens))]
    (<= odds 1)))
(map can-permute-palindrome ["code" "aab" "carerac"])
;;293
(defn generate-possible-next-moves [current-state]
  (letfn [(generate [start]
            (let [offset (.indexOf (subs current-state start) "++")
                  absolute-offset (+ start offset)
                  get-move (fn [offset]
                             (let [part1 (subs current-state 0 (+ start offset))
                                   part2 (subs current-state (+ absolute-offset 2))]
                               (str part1 "--" part2)))]

              (if (or (> start (- (count current-state) 2)) (= -1 offset))
                []
                (concat [(get-move offset)] (generate (inc absolute-offset))))))]
    (generate 0)))
(map generate-possible-next-moves ["++++" "+"])
(map-indexed + (range 10 21))

;;408 TODO
(defn valid-word-abbreviation [word abbr]
  (let [parts (vec (re-seq #"\d+" abbr))
        non-leading-zero? (fn [result digits] (if (= (first (vec digits)) \0)
                                                (reduced false)
                                                true))
        without-leading-zero (reduce non-leading-zero? true parts)
        word-letters (vec word)
        expanded-word (str/replace abbr #"\d+" #(str/join "" (take (Integer/parseInt %) (repeat "0"))))
        expanded-word-letters (vec expanded-word)
        compare-non-digits (fn [result i]
                              (if (= (expanded-word-letters i) \0)
                                true
                                (= (word-letters i) (expanded-word-letters i))))
        equal-non-digits (reduce compare-non-digits true (range (count expanded-word-letters)))]
    (and without-leading-zero (= (count word) (count expanded-word)) equal-non-digits)))
(map (partial apply valid-word-abbreviation) ['("internationalization" "i12iz4n") '("apple" "a2e")])

;;422  ;;GOOD
(defn valid-word-square [words]
  (let [padding (fn [s len]
                  (let [l (count s)]
                    (if (< l len)
                      (str s (apply str (repeat (- len l) "x")))
                      s)))
        m (mapv #(vec (padding % (count words))) words)
        transpose (fn [m]
                    (let [rows (count m)
                          cols (count (m 0))
                          mt (make-array Character/TYPE cols rows)
                          indices (for [r (range rows) c (range cols)] [r c])]
                      (doseq [[r c] indices]
                        (aset mt c r ((m r) c)))
                      (mapv vec mt)))]
    (if (not= (count m) (count (m 0)))
      false
      (= m (transpose m)))))
(map valid-word-square [["abcd","bnrt","crmy","dtye"] ["abcd","bnrt","crm","dt"] ["ball","area","read","lady"]])

;;734 ;;GOOD
(defn are-sentences-similar [sentence1 sentence2 similar-pairs]
  (let [similar-map (reduce (fn [m [w1 w2]]
                              (assoc m w1 w2)) {} (into similar-pairs (mapv reverse similar-pairs)))
        similar? (fn [i]
                   (let [w1 (sentence1 i)
                         w2 (sentence2 i)]
                     (or (= w1 w2) (= w1 (similar-map w2)))))]
    (and (= (count sentence1) (count sentence2))
      (every? similar? (range (count sentence1))))
    ))
(map (partial apply are-sentences-similar) ['(["great" "acting" "skills"] ["fine" "drama" "talent"] [["great" "fine"] ["drama" "acting"] ["skills" "talent"]])
                                            '(["great"] ["great"] [])
                                            '(["great"] ["doubleplus" "good"] [["great" "doubleplus"]])])
;;760
(defn anagram-mappings [nums1 nums2]
  (let [add-index (fn [m [index num]]
                    (assoc m num (conj (or (get m num) []) index)))
        indices-map (reduce add-index {} (map-indexed vector nums2))
        add (fn [[results m] num]
              (let [indices (get m num)]
                [(conj results (first indices)) (assoc m num (rest indices))]
                )
              )
        ]
    (first (reduce add [[] indices-map] nums1))))
(map (partial apply anagram-mappings) ['([12 28 46 32 50] [50 12 32 46 28]) '([84 46] [84 46])
                                       '([84 46 46] [84 46 46])])

;;800 ;;GOOD ;;DIFFICULT
(defn similar-rgb [color]
  (let [hex->int (fn [h]
                   (if (Character/isDigit h)
                     (- (int h) (int \0))
                     (+ (- (int h) (int \a)) 10)))
        int->hex (fn [n]
                   (if (< n 10)
                     (char (+ (int \0) n))
                     (char (- (+ n (int \a)) 10))
                   ))
        ->digits  (fn [color] (mapv hex->int (subs color 1)))
        square (fn [n] (* n n))
        from-digits' (fn [ds]
                      (+ (* 16 (ds 0)) (ds 1)))
        from-digits (fn [digits i]
                     (from-digits' (subvec digits i (+ i 2))))
        similarity' (fn [digits1 digits2 i]
                     (square (- (from-digits digits1 i) (from-digits digits2 i))))
        similarity (fn [digits1 digits2]
                     (reduce (fn [result index]
                               (- result  (similarity' digits1 digits2 index))) 0 (range 0 6 2)))
        digits (->digits color)
        rgbs (for [r (range 0 16) g (range 0 16) b (range 0 16)]
      [r r g g b b])]
    (->> (map (fn [rgb] [rgb (similarity digits rgb)]) rgbs)
         (sort (fn [l r] (compare (last r) (last l))))
         (first)
         (first)
         (map int->hex)
         (apply str)
         (#(str "#" %)))))
(map similar-rgb ["#09f166" "#4e3fe1"])

;;1056 ;;GOOD
(defn confusing-number [n]
  (letfn [(integer-digits' [n]
            (if (< n 10)
              [n]
              (conj (integer-digits' (quot n 10)) (rem n 10))))
          (integer-digits [n]
            (if (zero? n)
              [0]
              (integer-digits' n)))
          (invalid? [n]
            (pos? (count (set/intersection (set (integer-digits n)) #{2 3 4 5 7}))))]
    (let [rotation-map {9 6, 6 9, 0 0, 1 1, 8 8}
          digits (integer-digits n)]
      (and (not (invalid? n))
           (not= (map #(get rotation-map %) digits) digits)
           ))))
(map confusing-number [6 89 11 25])

;;1064
(defn fixed-point1 [arr]
  (reduce (fn [result index]
            (if (= index (arr index))
              (reduced index)
              result))
          -1 (range (count arr))))

(defn fixed-point [arr]
  (letfn [(find-fixed-point [nums start end]
            (let [middle (quot (+ start end) 2)]
              (cond
                (> start end) -1
                (< middle (nums middle)) (find-fixed-point nums start (dec middle))
                (> middle (nums middle)) (find-fixed-point nums (inc middle) end)
                :else middle)
              )
            )]
    (find-fixed-point arr 0 (dec (count arr)))
  ))
(map fixed-point [[-10 -5 0 3 7] [0 2 5 8 17] [-10 -5 3 4 7 9]])

;;1065
(defn index-pairs [text words]
  (letfn [(positions [haystack needle start]
            (let [len (count needle)
                  index (.indexOf (subs haystack start) needle)
                  absolute-index (+ start index)
                  ]
                (cond
                  (= index -1) []
                  (= (inc absolute-index) (count haystack)) [absolute-index (+ (dec absolute-index) len)]
                  :else (conj (positions haystack needle (inc absolute-index))
                              [absolute-index (+ (dec absolute-index) len)]))))
          (add-index-pairs [results word]
            (let [index-pairs (positions text word 0)]
              (if (empty? index-pairs)
                results
                (concat results index-pairs))))
          (compare-index-pair [l r]
            (if (= (first l) (first r))
              (compare (last l) (last r))
              (compare (first l) (first r))))]
    (->> (reduce add-index-pairs [] words)
         (sort compare-index-pair))))
(map (partial apply index-pairs) ['("thestoryofleetcodeandme" ["story" "fleet" "leetcode"])
                                  '("ababa" ["aba" "ab"])])

;;1085
(defn sum-of-digits [nums]
  (letfn [(integer-digits [n]
            (if (< n 10)
              [n]
              (conj (integer-digits (quot n 10)) (rem n 10))
              )
            )]
    (->> (apply min nums)
         (integer-digits)
         (apply +)
         (#(rem (inc %) 2)))))
(map sum-of-digits [[34 23 1 24 75 33 54 8] [99 77 33 66 55]])

;;1086
(defn high-five [items]
  (let [add (fn [result [id score]]
              (let [scores (or (get result id) [])]
                  (assoc result id (conj scores score))))
        student-scores (reduce add {} items)
        top-5-average (fn [[id scores]]
                        (let [tops (take 5 (sort > scores))
                              average (quot (apply + tops) (count tops))]
                          [id average]))
        compare-id (fn [l r] (compare (first l) (first r)))]
    (->> (map top-5-average student-scores)
         (sort compare-id))))
(map high-five [[[1 91] [1 92] [2 93] [2 97] [1 60] [2 77] [1 65] [1 87] [1 100] [2 100] [2 76]]
                [[1 100] [7 100] [1 100] [7 100] [1 100] [7 100] [1 100] [7 100] [1 100] [7 100]]])

;;1099
(defn two-sum-less-than-k1 [nums k]
  (let [len (count nums)
        sums (for [i (range (dec len)) j (range (inc i) len)]
               (+ (nums i) (nums j)))
        max-sums (filter #(< % k) sums)]
    (if (empty? max-sums)
      -1
      (apply max max-sums))))

(defn two-sum-less-than-k [nums k]
  (let [check-sum (fn [target [result index-map] num]
                    (if (nil? (get index-map num))
                      [false (assoc index-map (- target num) num)]
                      (reduced [true index-map])))
        two-sum (fn [nums target]
                  (first (reduce (partial check-sum target) [false {}] nums)))]
    (reduce (fn [sum k]
              (if (two-sum nums k)
                (reduced k)
                sum)) -1 (reverse (range 1 k)))))
(map (partial apply two-sum-less-than-k) ['([34 23 1 24 75 33 54 8] 60) '([10 20 30] 15)])

;;1118
(defn number-of-days [year month]
  (let [divisible? (fn [a b] (zero? (rem a b)))
        leap-year? (fn [year]
                     (let [rule1 #(and (divisible? year 4) (not (divisible? year 100)))
                           rule2 #(divisible? year 400)]
                     (or (rule1) (rule2))))
        month-days [0 31 28 31 30 31 30 31 31 30 31 30 31]
        extra-day (if (and (leap-year? year) (= month 2))
                    1
                    0)]
    (+ (month-days month) extra-day)))
(map (partial apply number-of-days) ['(1992 7) '(2000 2) '(1900 2)])

;;1119
(defn remove-vowels1 [s]
  (str/replace s #"[aeiou]" ""))
(defn remove-vowels [s]
  (let [vowels (set (seq "aeiou"))
        non-vowel? (fn [c] (not (contains?  vowels c)))]
   (str/join "" (filter non-vowel? (vec s)))))
(map remove-vowels ["leetcodeisacommunityforcoders" "aeiou"])

;;1133
(defn largest-unique-number [nums]
  (let [freqs (frequencies nums)
        unique? (fn [[num cnt]] (= cnt 1))
        unique-nums (map first (filter unique? freqs))]
    (if (empty? unique-nums)
      -1
      (apply max unique-nums))))
(map largest-unique-number [[5,7,3,9,4,9,8,3,1] [9,9,8,8]])

;;1134
(defn is-armstrong [n]
  (letfn [(integer-digits [n]
                           (if (< n 10)
                             [n]
                             (conj (integer-digits (quot n 10)) (rem n 10))))
          (power [n k]
            (reduce (fn [p _] (* p n)) 1 (range k)))]
    (let [digits (integer-digits n)
          len (count digits)]
      (->> (map #(power %1 len) digits)
           (apply +)
           (#(= % n))))))
(map is-armstrong [153 123])

;;1150
(defn is-majority-element1 [nums target]
  (let [freqs (frequencies nums)
        len (count nums)
        cnt (or (get freqs target) 0)
        ]
    (> cnt (quot len 2))))
(defn is-majority-element [nums target]
  (let [len (count nums)
        left (.indexOf nums target)
        right (.lastIndexOf nums target)]
    (>= (- right left) (quot len 2))))
(map (partial apply is-majority-element) ['([2 4 5 5 5 5 5 6 6] 5) '([10 100 101 101] 101)])

;;1165
(defn calculate-time [keyboard word]
  (let [abs (fn [n] (if (neg? n) (- 0 n) n))
        keyboard-map (into {} (map vector (vec keyboard) (range (count keyboard))))
        get-time (fn [c1 c2]
                   (let [index1 (get keyboard-map c1)
                         index2 (get keyboard-map c2)]
                     (abs (- index1 index2))))
        cs (vec word)
        add-time (fn [sum i]
                   (let [time (get-time (cs i) (cs (dec i)))]
                     (+ sum time)))
        initial (get keyboard-map (first cs))]
    (reduce add-time initial (range 1 (count cs)))))
(map (partial apply calculate-time) ['("abcdefghijklmnopqrstuvwxyz" "cba") '("pqrstuvwxyzabcdefghijklmno" "leetcode")])

;;1176
(defn diet-plan-performance [calories k lower upper]
  (let [len (count calories)
        get-point (fn [i]
                    (let [sum (apply + (subvec calories i (+ i k)))]
                      (cond
                        (< sum lower) -1
                        (> sum upper) 1
                        :else 0)))
        ]
  (apply + (map get-point (range (- (inc len) k))))))
(map (partial apply diet-plan-performance) ['([1 2 3 4 5] 1 3 3) '([3 2] 2 0 1) '([6 5 0 0] 2 1 5)])

;;1180
(defn count-letters [s]
  (let [count-distinct-substrings (fn [s] (let [n (count s)]
                                            (quot (* (inc n) n) 2)))
        cs (vec s)
        add-str (fn [[results result] i]
                  (cond
                    (= i (count cs)) [(conj results result) []]
                    (= (last result) (cs i)) [results (conj result (cs i))]
                    :else [(conj results result) [(cs i)]]))
        parts (first (reduce add-str [[] [(cs 0)]] (range 1 (inc (count s)))))]
    (apply + (map count-distinct-substrings parts))
    ))
(map count-letters ["aaaba" "aaaaaaaaaa"])

;;1196
(defn max-number-of-apples [weight]
  (let [add (fn [[result sum] w]
              (if (< (+ sum w) 5000)
                [(inc result) (+ sum w)]
                (reduced [result sum])
              ))]
 (first (reduce add [0 0] (sort weight)))))
(map max-number-of-apples [[100 200 150 1000] [900 950 800 1000 700 800]])

;;1213
(defn array-intersection [arr1 arr2 arr3]
  (->> (map set [arr1 arr2 arr3])
       (apply set/intersection)
       (sort)
       (vec)))
(map (partial apply array-intersection) ['([1 2 3 4 5] [1 2 5 7 9] [1 3 4 5 8]) '([197 418 523 876 1356] [501 880 1593 1710 1870] [521 682 1337 1395 1764])])

;;1228
(defn missing-number [arr]
  (let [len (count arr)
        a (first arr)
        b (last arr)
        sum (quot (* (+ a b) (inc len)) 2)
        sum' (apply + arr)]
    (- sum sum')))
(map missing-number [[5 7 11 13] [15 13 12]])

;;1243
(defn transform-array [arr]
  (letfn [(transform [xs]
            (let [change (fn [i]
                           (let [x0 (xs (dec i))
                                 x1 (xs i)
                                 x2 (xs (inc i))]
                             (cond
                               (and (< x1 x0) (< x1 x2)) (inc x1)
                               (and (> x1 x0) (> x1 x2)) (dec x1)
                               :else x1)))
                  result' (map change (range 1 (dec (count xs))))
                  result (vec (concat [(first xs)] result' [(last xs)]))]
              (if (= result xs)
                result
                (transform result))))]
    (transform arr)))
(map transform-array [[6 2 3 4] [1 6 3 4 3 5]])

;;1271
(defn to-hexspeak [num]
  (letfn [(->hex' [n]
            (if (< n 16)
              [n]
              (conj (->hex' (quot n 16)) (rem n 16))))
          (->hex [n]
            (if (zero? n)
              [0]
              (->hex' n))
            )
          (invalid? [digits] (not-empty (filter #(and (<= % 9) (>= % 2)) digits)))
          (translate [digit] (case digit
                               1 \I
                               0 \O
                               (char (+ (int \A) (- digit 10)))))
          (->hexspeak [digits] (str/join "" (map translate digits)))]
    (let [digits (->hex (Integer/parseInt num))]
      (if (invalid? digits)
        "ERROR"
        (->hexspeak digits)))))
(map to-hexspeak ["257" "3"])

;; ;;1426
(defn count-elements [arr]
  (let [freqs (frequencies arr)
        nums (drop-last (sort (keys freqs)))
        count-num (fn [result num]
                    (let [cnt (get freqs num)]
                      (if (nil? (get freqs (inc num)))
                        result
                      (+ result cnt))))]
    (reduce count-num 0 nums)))
(map count-elements [[1 2 3] [1 1 3 3 5 5 7 7] [1 3 2 3 5 0] [1 1 2 2] [1 1 2]])

;;1427
(defn string-shift [s shift]
  (let [left-shift (fn [s amount]
                     (str (subs s amount) (subs s 0 amount)))
        right-shift (fn [s amount]
                      (let [len (count s)]
                        (str (subs s (- len amount)) (subs s 0 (- len amount)))))
        string-shift' (fn [s [direction amount]]
                        (if (zero? direction)
                          (left-shift s amount)
                          (right-shift s amount)))
        ]
    (reduce string-shift' s shift)))
(map (partial apply string-shift) ['("abc" [[0,1],[1,2]]) '("abcdefg" [[1,1],[1,1],[0,2],[1,3]])])

;;1708
(defn largest-subarray [nums k]
  (let [max-value (apply max (drop-last (dec k) nums))
        index (.indexOf nums max-value)]
  (subvec nums index (+ index k))))
(map (partial apply largest-subarray) ['([1 4 5 2 3] 3) '([1 4 5 2 3] 4) '([1 4 5 2 3] 1)])

;;1826 ;;GOOD
(defn bad-sensor [sensor1 sensor2]
  (let [compare-sensor (fn [defect-sensor good-sensor]
                         (let [data (drop-last defect-sensor)]
                           (reduce (fn [result i]
                                     (if (= data (concat (subvec good-sensor 0 i) (subvec good-sensor (inc i))))
                                       (reduced true)
                                       result))
                                   false (range (count good-sensor)))))
        both-correct (= sensor1 sensor2)
        first-incorrect (compare-sensor sensor1 sensor2)
        second-incorrect (compare-sensor sensor2 sensor1)]
    (cond
      (and first-incorrect (not both-correct) (not second-incorrect)) 1
      (and second-incorrect (not both-correct) (not first-incorrect)) 2
      :else -1)))
(map (partial apply bad-sensor) ['([2 3 4 5] [2 1 3 4]) '([2 2 2 2 2] [2 2 2 2 5]) '([2 3 2 2 3 2] [2 3 2 3 2 7])])

(defn is-decomposable [s]
  (let [cs (vec s)
        len (count s)
        add (fn [[results result] i]
              (cond
                (= i len) [(conj results result) []]
                (= (last result) (cs i)) (if (= 3 (count result))
                                           [(conj results result) [(cs i)]]
                                           [results (conj result (cs i))])
                :else [(conj results result) [(cs i)]]))
        split-into-subarrays (fn [s]
                               (let [rs (range 1 (inc len))]
                                 (first (reduce add [[] [(first cs)]] rs))))
        parts (split-into-subarrays s)
        decomposible? (fn [freqs]
                        (and (= (sort (keys freqs)) [2 3])
                             (= 1 (get freqs 2))
                             (not (nil? (get freqs 3)))))
        ]
    (if (empty? parts)
      false
      (->> (map count parts)
           (frequencies)
           (decomposible?)
           ))))
(map is-decomposable ["000111000" "00011111222" "011100022233"])
;;2021-10-13

;;1863
(defn subset-xor-sum [nums]
  (let [len (count nums)]
    (for [i (range (dec len)) j (range (inc i) len)]
      )
))
(map subset-xor-sum [[1 3] [5 1 6] [3 4 5 6 7 8]])
