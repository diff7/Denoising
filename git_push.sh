#echo "Enter your message"
#read message
git add .
git commit -m"auto push"
if [ -n "$(git status - porcelain)" ];
then
	 echo "IT IS CLEAN"
 else
	  git status
	   echo "Pushing data to origin ..."
	    git push -u origin master
    fi

