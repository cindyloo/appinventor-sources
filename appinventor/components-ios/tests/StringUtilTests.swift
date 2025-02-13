// -*- mode: swift; swift-mode:basic-offset: 2; -*-
// Copyright 2018-2023 MIT, All rights reserved
// Released under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0

import Foundation
import XCTest
@testable import AIComponentKit

class StringUtilTests: XCTestCase {
  func testJoinStrings() throws {
    XCTAssertEqual("a:b:c",
                   StringUtil.joinStrings(["a", "b", "c"], NSString(string: ":").description as NSString))
  }

  /*func testReverseString() throws {
    XCTAssertEqual(["dlroW"],
                   //try StringUtil.reverseString("World"))
  }*/

  func testReplaceMappingsWithLongestString()  {
    XCTAssertEqual("peace for you and me and ourselves please",
                   StringUtil.replaceAllMappingsLongestString("\"peace for me and you and yourselves please\"", ["you":"me", "me": "you","I": "you", "you": "I", "yourselves": "ourselves", "ourselves": "yourselves","myself": "yourself", "yourself": "myself", "mine": "yours", "ours": "yours", "yours": "mine" ]))// repl issue vs  "yours": "ours"]))
  }

  func testReplaceMappingsWithDictionary()  {
    XCTAssertEqual("you me peace please",
                   StringUtil.replaceAllMappingsLongestString("\"me you peace please\"", ["you":"me", "me": "you"]))
  }

  


}
