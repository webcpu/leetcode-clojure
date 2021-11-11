;;#!/usr/bin/env bb
(->> (slurp "./leetcode.clj")
     (#(clojure.string/split % #"\n"))
     (map (fn [s] (first (re-seq #";;\d+" s))))
     (flatten)
     (filter (comp not nil?))
     (map (fn [s] (re-seq #"\d+" s)))
     (count)
     )

;; (let [max-apple-kids 28 ;; 3*9 + 1
;;       count-apple-kids (fn [n] (range 0 n 2))
;;       count-orange-kids (fn [n] (range (dec n) -1 -4))
;;       count-kids (fn [max-kids i] (let [apple-kids (count-apple-kids i)
;;                                         orange-kids (count-orange-kids i)
;;                                         fruit-kids (concat apple-kids orange-kids)]
;;                                     (if (= 10 (count (set fruit-kids)))
;;                                            (reduced i)
;;                                            max-kids)))]
;;     (reduce count-kids 0 (range max-apple-kids 0 -1)))

;; (letfn [(f [n]
;;           (if (zero? n)
;;             1
;;             (- n (m (f (dec 1))))))
;;         (m [n]
;;           (if (zero? n)
;;             0
;;             (- n (f (m (dec 1))))))]

;;   (defn F [n]
;;      (if (zero? n) 1 n))
;;   (map f (range 100000 100001)
;;        ))

