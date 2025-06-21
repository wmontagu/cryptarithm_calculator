This is a cryptarithm calculator (mathematical puzzles that are represnted in verbal text, that must be decoded into numbers). My experiences with cryptarithms started in high school, where I solve many in my free time, and took a class that I had to do them in. In my experience, they always felt fun until they are not, which is why I implemented it.


These are two different ways I implemented it (I called it Easy and Medium). One of these methods was the simple brute force methods, where I try every possible case after applying some basic constraints of the question. The second method is something used much of the time in Constraint Satisfaction Problems (CSP), using Forward Checking to and MRV (Minimal Remaining Value) hueristic to understand what variable to select next.

In the future, I hope to an ML algorithm such that it can view the question, to either confirm or deny (specifically only from structure) to access weather the item is a valid cryptarithm or not.

I can improve my Medium by using AC-3 (Arc Consistency), by allowing a queuing system... This will also happen in the future.