/////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2002 - 2022,
//  All rights reserved.
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//     1. Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//     2. Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//     3. The names of its contributors may not be used to endorse or promote
//        products derived from this software without specific prior written
//        permission.
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "SePreDefine.h"

SE_NAMESPACE_BEGIN

/*************************************************************************
****************************    SeArray2D    *****************************
*************************************************************************/

//!	@brief	��ά�����������������ά��������������ԣ��С��С�����һ��(����)�洢��ά����ֵ��������
// Ŀǰ��������������Ŀ�Ļ�������ϡ��ṹ�Ĵ洢
//�����������ǽ�һ����ά������/�е���ʽ��ƽ��һά��������ʽ���д洢����ߵ�typeһ����int/float

template<typename Type> class SeArray2D
{

public:

	//!	@brief	Default constructor.
	SeArray2D() : m_Rows(0), m_Columns(0) {}

	//!	@brief	Construct and resize.Ĭ�ϸ�����m_value��ֵΪ0
	explicit SeArray2D(size_t _Rows, size_t _Columns) : m_Rows(_Rows), m_Columns(_Columns), m_Values(_Rows * _Columns) {} //Ĭ�ϸ�ֵΪ0

	//!	@brief	Construct, resize and memory set.
	//m_Values(_Rows * _Columns, _Value)����������std::vector<Type> m_Values(_Rows * _Columns, _Value);����һ������Ϊ_Rows * _Columns, ֵ_Value������

	//ע�����︳ֵ��value�����Ͳ������������������ͣ�����int������
	explicit SeArray2D(size_t _Rows, size_t _Columns, Type _Value) : m_Rows(_Rows), m_Columns(_Columns), m_Values(_Rows * _Columns, _Value) {}

public:

	//! @brief  Data will be reserved in Column Major format �������洢
	void Resize(size_t _Rows, size_t _Columns)
	{
		m_Values.resize(_Rows * _Columns);

		m_Columns = _Columns;

		m_Rows = _Rows;
	}

	//! @brief  release spare memory
	void ShrinkToFit()
	{
		m_Values.shrink_to_fit(); //�ͷſ����ڴ棬��size&capacity����һ��
	}

	//!	@brief	Fill data. ������䣬����ÿ��Ԫ�ض���ͬһ��ֵ���
	void Memset(Type _Value, size_t _Begin = 0, size_t _End = SIZE_MAX)
	{
		_End = _End < m_Values.size() ? _End : m_Values.size(); //ָ��size�������

		for (size_t i = _Begin; i < _End; ++i)
		{
			m_Values[i] = _Value;
		}
	}

	//!	@brief	Exchange context with right. 
	void Swap(SeArray2D & _Right)
	{
		m_Values.swap(_Right.m_Values); //����������size��capacity��value
		 
		SE_SWAP(m_Columns, _Right.m_Columns); //��Ӧ�ṹ���ڵ�������ϢҲ��������

		SE_SWAP(m_Rows, _Right.m_Rows);
	}

	void Clear()
	{
		m_Values.clear();

		m_Values.shrink_to_fit(); //����Ԫ����������ͷ��ڴ棬������ϢҲ���=0

		m_Rows = m_Columns = 0;
	}

public:

	const Type * operator[](size_t i) const { SE_ASSERT(i < m_Rows);  return &m_Values[m_Columns * i]; } //[]��������أ����ص�i��Ԫ�صĵ�ַ  A[i][j]��Ӧi��j�е�Ԫ�أ�Ҳ����������i*m_columns+j��Ԫ��

	Type * operator[](size_t i) { SE_ASSERT(i < m_Rows);  return &m_Values[m_Columns * i]; }

	const Type * Ptr() const { return m_Values.data(); } //���ص�һ��Ԫ��ֵ(&��ַ)����֧���޸�

	bool IsEmpty() const { return m_Values.empty(); } //�ж��Ƿ�Ϊ��

	size_t Size() const { return m_Values.size(); }//��������s����

	size_t Capacity() const { return m_Values.capacity(); } //�����ѷ����ڴ��С��Ҳ���Ǹýṹ��Ĵ�С����=�����洢�Ĵ�С������

	size_t Columns() const { return m_Columns; }

	size_t Rows() const { return m_Rows; }

	Type * Ptr() { return m_Values.data(); } //���ص�ǰ�����ĵ�һ��Ԫ�ص�ַ����֧���޸�:ʵ�ʾ��Ƿ���m_values�������

private:

	std::vector<Type> m_Values; //һά������Ԫ������Ϊtype���ͣ�����������ʽ

	size_t m_Rows, m_Columns;   //�޷������͵ı��� 
};

SE_NAMESPACE_END