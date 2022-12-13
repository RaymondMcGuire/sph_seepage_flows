/***
 * @Author: Daniel Illescas Romero
 * @Date: 2022-12-13 21:11:24
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-12-13 21:21:10
 * @FilePath: \sph_seepage_flows\extlib\include\file_io.hpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#pragma once

#include <optional>
#include <fstream>
#include <string>
#include <stdexcept>
#include <memory>

namespace evt
{

	class BaseFileIO
	{
	public:
		struct OpenCloseException : public std::runtime_error
		{
			OpenCloseException() : runtime_error("error") {}
			OpenCloseException(const std::string &message) : runtime_error(message.c_str()) {}
		};

	protected:
		std::fstream fileStream;
		std::ios_base::openmode inputOutputMode;
		std::string fileName_;

		void open(const std::ios_base::openmode inputOutputMode) noexcept(false)
		{

			if (this->inputOutputMode == inputOutputMode)
			{
				return;
			}

			close();
			this->inputOutputMode = inputOutputMode;
			fileStream.open(fileName_, inputOutputMode);

			if (fileStream.fail())
			{
				throw OpenCloseException("File couldn't be open");
			}
		}

		BaseFileIO(const std::string &fileName)
		{
			this->fileName_ = fileName;
		}

	public:
		static bool exists(const std::string &filePath)
		{
			std::ifstream file(filePath);
			return !file.fail();
		}

		void seekInputPosition(std::size_t offsetPosition, std::ios_base::seekdir position = std::ios::beg)
		{
			fileStream.seekg(offsetPosition, position);
		}

		void open(const std::string &fileName)
		{
			this->fileName_ = fileName;
			this->close();
		}

		bool endOfFile() const
		{
			return fileStream.eof();
		}

		std::string fileName() const noexcept
		{
			return fileName_;
		}

		/// Optional to use, the class automatically closes the file
		void close()
		{
			if (fileStream.is_open())
			{
				fileStream.close();
			}
		}

		~BaseFileIO()
		{
			close();
		}
	};

	class BinaryFileIO : public BaseFileIO
	{
	public:
		BinaryFileIO(const std::string &fileName) : BaseFileIO(fileName) {}

		template <typename Type>
		void write(Type &&content, bool appendContent = false)
		{

			if (appendContent)
			{
				open(std::ios::binary | std::ios::out | std::ios::in | std::ios::app);
			}
			else
			{
				open(std::ios::binary | std::ios::out | std::ios::in | std::ios::trunc);
			}

			if constexpr (!std::is_same<Type, std::string>())
			{
				fileStream.write(reinterpret_cast<char *>(&content), sizeof(content));
			}
			else
			{
				fileStream.write(content.c_str(), content.length());
			}
		}

		template <typename Type, typename = typename std::enable_if<!std::is_same<Type, std::string>::value>::type>
		Type readWithOffset(std::size_t offset)
		{

			Type readInput{};
			open(std::ios::in | std::ios::binary);
			if (offset > 0)
			{
				seekPosition(offset);
			}
			fileStream.read(reinterpret_cast<char *>(&readInput), sizeof(readInput));
			return readInput;
		}

		template <typename Type, typename = typename std::enable_if<!std::is_same<Type, std::string>::value>::type>
		Type read()
		{
			return readWithOffset<Type>(0);
		}

		template <typename Type, typename = typename std::enable_if<std::is_same<Type, std::string>::value>::type>
		Type readWithOffset(std::size_t size, std::size_t offset)
		{

			std::string readContent;
			open(std::ios::in);
			if (offset > 0)
			{
				seekInputPosition(offset);
			}
			fileStream >> readContent;

			return readContent;
		}

		template <typename Type, typename = typename std::enable_if<std::is_same<Type, std::string>::value>::type>
		Type read(std::size_t size)
		{
			return readWithOffset<Type>(size, 0);
		}

		void seekPosition(std::size_t offsetPosition, std::ios_base::seekdir position = std::ios::beg)
		{
			fileStream.seekp(offsetPosition, position);
		}
	};

	class PlainTextFileIO : private BaseFileIO
	{
	public:
		PlainTextFileIO(const std::string &fileName) : BaseFileIO(fileName) {}

		template <typename Type>
		void write(const Type &contentToWrite, bool appendContent = false)
		{

			if (appendContent)
			{
				open(std::ios::out | std::ios::in | std::ios::app);
			}
			else
			{
				open(std::ios::out | std::ios::in | std::ios::trunc);
			}

			fileStream << contentToWrite;
		}

		/// Reads text content word by word
		std::string readWithOffset(std::size_t offset = 0)
		{

			std::string readContent;
			open(std::ios::in);
			if (offset > 0)
			{
				seekInputPosition(offset);
			}
			fileStream >> readContent;

			return readContent;
		}

		/// Reads text content word by word
		std::string read()
		{
			return readWithOffset(0);
		}

		std::string getline()
		{

			std::string s;

			open(std::ios::in);
			std::getline(fileStream, s);

			return s;
		}

		std::optional<std::string> safeRead()
		{

			std::string readContent;
			open(std::ios::in);
			fileStream >> readContent;

			if ((fileStream.eof() && readContent.empty()) || fileStream.fail())
			{
				return std::nullopt;
			}

			return readContent;
		}

		std::optional<std::string> safeGetline()
		{

			std::string s;

			open(std::ios::in);
			std::getline(fileStream, s);

			if ((fileStream.eof() && s.empty()) || fileStream.fail())
			{
				return std::nullopt;
			}

			return s;
		}

		std::string toString()
		{

			this->open(std::ios::in);

			std::string line;
			std::string outputContent;

			while (std::getline(fileStream, line))
			{
				outputContent += line + '\n';
			}

			return outputContent;
		}

		static std::string toString(const std::string &fileName)
		{
			return PlainTextFileIO(fileName).toString();
		}

		static void saveTextTo(const std::string &fileName, const std::string &text)
		{
			PlainTextFileIO fileToWrite(fileName);
			fileToWrite.open(std::ios::out);
			fileToWrite.write(text);
		}
	};

	struct FileIO
	{
		static PlainTextFileIO plainText(const std::string &fileName)
		{
			return PlainTextFileIO(fileName);
		}
		static BinaryFileIO binary(const std::string &fileName)
		{
			return BinaryFileIO(fileName);
		}
	};
}
