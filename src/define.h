#ifndef __DEFINE_H__
#define __DEFINE_H__
//stadard include

//std
#include <memory>
#include <iostream>


#define		IF_NOT(x)		if( (!(x)) ? (std::cout << "*IF_NOT(" << #x << ")* , " << __FILE__ << "(" << __LINE__ << ")"), 1 : 0 )


#define CHECK(val) if (!val) { \
		std::cout << __FILE__ << "("<< __LINE__ << ")CHECK fail";	\
		exit(0);			\
	}


#define CHECK_EQ(val1, val2) if (val1 != val2) { \
		std::cout << __FILE__ << "("<< __LINE__ << ")CHECK_EQ fail";	\
		exit(0);			\
	}

#endif //__DEFINE_H__
