# Gradient-boosting learning for multi-class classification.

This is a simple library that implements simple versions of gradient boosting
and xgboost for classification, in pure Go. 

I have tried to keep things simple, including the data format.

There is some parallelization/concurrency for the cross-validation grid for XGBoost, I should
add it to the regular gradient boosting CV grid soon. That's as far as I'll got for now.

Some facilities, such as cross-validation and file-reading (a _very_ naive/incomplete
reader for the libSVM format) are provided.

The objective is for this library to become a part of goChem when its mature enough. Still, there is nothing specific about Chemistry in this package.

Both the regular gradient-boosting as well as the xgboost implementations are close ports/translations from the following Python implementations:

* [Gradient boosting multi-class classification from scratch](https://randomrealizations.com/posts/gradient-boosting-multi-class-classification-from-scratch/)
* [XGBoost from scratch](https://randomrealizations.com/posts/xgboost-from-scratch/)

Both by [Matt Bowers](https://github.com/mcb00)

Disclaimer: The whole thing is not really "production quality" as of now. More debugging, a lot more test and cleanup are needed.


(c) 2024 Raul Mera A.

This program, including its documentation, 
is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation; either version 2.1 of the 
License, or (at your option) any later version.
          
This program and its documentation is distributed in the hope that 
it will be useful, but WITHOUT ANY WARRANTY; without even the 
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE.  See the GNU General Public License for more details.
                    
You should have received a copy of the GNU Lesser General 
Public License along with this program. If not, see 
<http://www.gnu.org/licenses/>. 

