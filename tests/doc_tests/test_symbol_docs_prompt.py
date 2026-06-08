# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Test all code snippets from prompt.py source documentation

RULES:
Use this consistent testing format:
- Each test function should correspond to a specific code snippet or section in the documentation.
- Include comments indicating the source documentation file and section for clarity.
- No extra tests beyond those needed to validate the snippets.

Copy the snippet from the source **exactly** as it appears in the documentation.
Surround the original source with:
##############################################################################
# From: src/apple_fm_sdk/<source_file>.py
# class, function, or other entity name: <source_section_name>
<actual code here uncommented>
##############################################################################

The test passes if the snippet runs without errors. No additional assertions are necessary
beyond ensuring the snippet executes successfully.
"""

import pytest
from pathlib import Path

# Get the path to test resources
TEST_RESOURCES_DIR = Path(__file__).parent.parent / "resources"
SIMPLE_IMAGE = TEST_RESOURCES_DIR / "test-simple-image.jpeg"
TEXT_DENSE_IMAGE = TEST_RESOURCES_DIR / "test-text-dense-image.png"

# =============================================================================
# IMAGE ATTACHMENT TESTS (from src/apple_fm_sdk/prompt.py)
# =============================================================================


@pytest.mark.asyncio
async def test_image_attachment_basic(model):
    """Test from: src/apple_fm_sdk/prompt.py - ImageAttachment class docstring - Basic image attachment"""
    print("\n=== Testing Basic Image Attachment ===")

    ##############################################################################
    # From: src/apple_fm_sdk/prompt.py
    # class, function, or other entity name: ImageAttachment - Basic image attachment
    # import apple_fm_sdk as fm
    # from pathlib import Path
    #
    # session = fm.LanguageModelSession()
    # image = fm.ImageAttachment(Path("photo.jpg"))
    # response = await session.respond(["What's in this image?", image])
    ##############################################################################

    # Actual test using real image path
    import apple_fm_sdk as fm

    try:
        session = fm.LanguageModelSession(model=model)
        image = fm.ImageAttachment(SIMPLE_IMAGE)
        response = await session.respond(["What's in this image?", image])
        assert response is not None
        print("✅ Basic image attachment - PASSED")
    except fm.ImagePromptError as e:
        pytest.skip(f"Image prompts not supported: {e}")


@pytest.mark.asyncio
async def test_image_attachment_labeled(model):
    """Test from: src/apple_fm_sdk/prompt.py - ImageAttachment class docstring - Labeled image attachment"""
    print("\n=== Testing Labeled Image Attachment ===")

    ##############################################################################
    # From: src/apple_fm_sdk/prompt.py
    # class, function, or other entity name: ImageAttachment - Labeled image attachment
    # import apple_fm_sdk as fm
    # from pathlib import Path
    #
    # session = fm.LanguageModelSession()
    # diagram = fm.ImageAttachment(Path("diagram.png"), label="architecture")
    # response = await session.respond(
    #     ["Explain the architecture shown in the diagram", diagram]
    # )
    ##############################################################################

    # Actual test using real image path
    import apple_fm_sdk as fm

    try:
        session = fm.LanguageModelSession(model=model)
        diagram = fm.ImageAttachment(SIMPLE_IMAGE, label="architecture")
        response = await session.respond(
            ["Explain the architecture shown in the diagram", diagram]
        )
        assert response is not None
        print("✅ Labeled image attachment - PASSED")
    except fm.ImagePromptError as e:
        pytest.skip(f"Image prompts not supported: {e}")


@pytest.mark.asyncio
async def test_image_attachment_multiple_with_labels(model):
    """Test from: src/apple_fm_sdk/prompt.py - ImageAttachment class docstring - Multiple images with labels"""
    print("\n=== Testing Multiple Images with Labels ===")

    ##############################################################################
    # From: src/apple_fm_sdk/prompt.py
    # class, function, or other entity name: ImageAttachment - Multiple images with labels
    # import apple_fm_sdk as fm
    # from pathlib import Path
    #
    # session = fm.LanguageModelSession()
    # before = fm.ImageAttachment(Path("before.jpg"), label="before")
    # after = fm.ImageAttachment(Path("after.jpg"), label="after")
    # response = await session.respond(
    #     ["Compare the before and after images", before, after]
    # )
    ##############################################################################

    # Actual test using real image paths
    import apple_fm_sdk as fm

    try:
        session = fm.LanguageModelSession(model=model)
        before = fm.ImageAttachment(SIMPLE_IMAGE, label="before")
        after = fm.ImageAttachment(TEXT_DENSE_IMAGE, label="after")
        response = await session.respond(
            ["Compare the before and after images", before, after]
        )
        assert response is not None
        print("✅ Multiple images with labels - PASSED")
    except fm.ImagePromptError as e:
        pytest.skip(f"Image prompts not supported: {e}")


@pytest.mark.asyncio
async def test_image_attachment_init_unlabeled():
    """Test from: src/apple_fm_sdk/prompt.py - ImageAttachment.__init__ - Create unlabeled attachment"""
    print("\n=== Testing ImageAttachment.__init__ Unlabeled ===")

    ##############################################################################
    # From: src/apple_fm_sdk/prompt.py
    # class, function, or other entity name: ImageAttachment.__init__ - Create an unlabeled image attachment
    # import apple_fm_sdk as fm
    # from pathlib import Path
    #
    # # Create an unlabeled image attachment
    # image = fm.ImageAttachment(Path("photo.jpg"))
    ##############################################################################

    # Actual test using real image path
    import apple_fm_sdk as fm

    image = fm.ImageAttachment(SIMPLE_IMAGE)
    assert image is not None
    print("✅ ImageAttachment.__init__ unlabeled - PASSED")


@pytest.mark.asyncio
async def test_image_attachment_init_labeled():
    """Test from: src/apple_fm_sdk/prompt.py - ImageAttachment.__init__ - Create labeled attachment"""
    print("\n=== Testing ImageAttachment.__init__ Labeled ===")

    ##############################################################################
    # From: src/apple_fm_sdk/prompt.py
    # class, function, or other entity name: ImageAttachment.__init__ - Create a labeled image attachment
    # import apple_fm_sdk as fm
    # from pathlib import Path
    #
    # # Create a labeled image attachment
    # diagram = fm.ImageAttachment(Path("diagram.png"), label="system_diagram")
    ##############################################################################

    # Actual test using real image path
    import apple_fm_sdk as fm

    diagram = fm.ImageAttachment(SIMPLE_IMAGE, label="system_diagram")
    assert diagram is not None
    assert diagram._label == "system_diagram"
    print("✅ ImageAttachment.__init__ labeled - PASSED")


# =============================================================================
# PROMPT TYPE ALIAS TESTS (from src/apple_fm_sdk/prompt.py)
# =============================================================================


@pytest.mark.asyncio
async def test_prompt_simple_text():
    """Test from: src/apple_fm_sdk/prompt.py - Prompt type alias - Simple text prompt"""
    print("\n=== Testing Simple Text Prompt ===")

    ##############################################################################
    # From: src/apple_fm_sdk/prompt.py
    # class, function, or other entity name: Prompt - Simple text prompt
    prompt = "Hello, how are you?"
    ##############################################################################

    assert prompt is not None
    assert isinstance(prompt, str)
    print("✅ Simple text prompt - PASSED")


@pytest.mark.asyncio
async def test_prompt_image_only():
    """Test from: src/apple_fm_sdk/prompt.py - Prompt type alias - Image-only prompt"""
    print("\n=== Testing Image-only Prompt ===")

    ##############################################################################
    # From: src/apple_fm_sdk/prompt.py
    # class, function, or other entity name: Prompt - Image-only prompt
    # from pathlib import Path
    #
    # prompt = ImageAttachment(Path("diagram.png"))
    ##############################################################################

    # Actual test using real image path
    from apple_fm_sdk.prompt import ImageAttachment

    prompt = ImageAttachment(SIMPLE_IMAGE)
    assert prompt is not None
    print("✅ Image-only prompt - PASSED")


@pytest.mark.asyncio
async def test_prompt_mixed_text_and_image():
    """Test from: src/apple_fm_sdk/prompt.py - Prompt type alias - Mixed text and image prompt"""
    print("\n=== Testing Mixed Text and Image Prompt ===")

    ##############################################################################
    # From: src/apple_fm_sdk/prompt.py
    # class, function, or other entity name: Prompt - Mixed text and image prompt
    # from pathlib import Path
    #
    # prompt = [
    #     "Analyze this diagram and explain the architecture:",
    #     ImageAttachment(Path("diagram.png"), label="system_architecture"),
    # ]
    ##############################################################################

    # Actual test using real image path
    from apple_fm_sdk.prompt import ImageAttachment

    prompt = [
        "Analyze this diagram and explain the architecture:",
        ImageAttachment(SIMPLE_IMAGE, label="system_architecture"),
    ]
    assert prompt is not None
    assert isinstance(prompt, list)
    assert len(prompt) == 2
    print("✅ Mixed text and image prompt - PASSED")


@pytest.mark.asyncio
async def test_prompt_multiple_images_with_text():
    """Test from: src/apple_fm_sdk/prompt.py - Prompt type alias - Multiple images with text"""
    print("\n=== Testing Multiple Images with Text Prompt ===")

    ##############################################################################
    # From: src/apple_fm_sdk/prompt.py
    # class, function, or other entity name: Prompt - Multiple images with text
    # from pathlib import Path
    #
    # prompt = [
    #     "Compare these two images:",
    #     ImageAttachment(Path("before.jpg"), label="before"),
    #     ImageAttachment(Path("after.jpg"), label="after"),
    #     "What are the main differences?",
    # ]
    ##############################################################################

    # Actual test using real image paths
    from apple_fm_sdk.prompt import ImageAttachment

    prompt = [
        "Compare these two images:",
        ImageAttachment(SIMPLE_IMAGE, label="before"),
        ImageAttachment(TEXT_DENSE_IMAGE, label="after"),
        "What are the main differences?",
    ]
    assert prompt is not None
    assert isinstance(prompt, list)
    assert len(prompt) == 4
    print("✅ Multiple images with text prompt - PASSED")


# =============================================================================
# ERROR HANDLING TESTS (from src/apple_fm_sdk/prompt.py)
# =============================================================================


@pytest.mark.asyncio
async def test_image_prompt_error_nonexistent_file():
    """Test from: src/apple_fm_sdk/prompt.py - ImagePromptError - Handling image prompt errors"""
    print("\n=== Testing ImagePromptError Handling ===")

    ##############################################################################
    # From: src/apple_fm_sdk/prompt.py
    # class, function, or other entity name: ImagePromptError - Handling image prompt errors
    import apple_fm_sdk as fm
    from pathlib import Path

    try:
        image = fm.ImageAttachment(Path("nonexistent.jpg"))
    except fm.ImagePromptError as e:
        print(f"Failed to create image attachment: {e}")
    ##############################################################################

    # Verify the error is raised
    with pytest.raises(fm.ImagePromptError):
        image = fm.ImageAttachment(Path("nonexistent.jpg"))

    print("✅ ImagePromptError handling - PASSED")


@pytest.mark.asyncio
async def test_image_prompt_error_in_session(model):
    """Test from: src/apple_fm_sdk/prompt.py - ImagePromptError - Error handling in session"""
    print("\n=== Testing ImagePromptError in Session ===")

    ##############################################################################
    # From: src/apple_fm_sdk/prompt.py
    # class, function, or other entity name: ImagePromptError - Error handling in session
    # import apple_fm_sdk as fm
    # from pathlib import Path
    #
    # try:
    #     session = fm.LanguageModelSession()
    #     response = await session.respond(
    #         ["Describe this image:", fm.ImageAttachment(Path("photo.jpg"))]
    #     )
    # except fm.ImagePromptError as e:
    #     print(f"Failed to process image prompt: {e}")
    ##############################################################################

    # Actual test using real image path - this should succeed
    import apple_fm_sdk as fm

    try:
        session = fm.LanguageModelSession(model=model)
        response = await session.respond(
            ["Describe this image:", fm.ImageAttachment(SIMPLE_IMAGE)]
        )
        assert response is not None
        print("✅ ImagePromptError in session - PASSED")
    except fm.ImagePromptError as e:
        pytest.skip(f"Image prompts not supported: {e}")
