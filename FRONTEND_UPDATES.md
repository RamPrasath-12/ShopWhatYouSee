# Frontend Updates - Photo Upload & Auto Video Feature

## Summary of Changes

The VideoPlayer component has been enhanced with the following features:

### 1. **Auto-Play Video on Component Mount**
- Video now automatically starts playing when the component loads (500ms delay)
- Uses `useEffect` hook to handle auto-play
- Handles browser auto-play policies gracefully with try-catch

### 2. **Photo Upload Functionality**
- Added a photo upload input that accepts all image formats (JPG, PNG, WebP, etc.)
- Users can switch between **Video Mode** and **Photo Mode** using button toggles
- Mode selector clearly indicates which input method is active

### 3. **Unified Processing Pipeline**
Both video frames and uploaded photos now go through the **identical processing pipeline**:

#### Processing Steps:
1. **YOLOv8 Detection** - Detects fashion items (clothing, accessories)
2. **AG-MAN Attribute Extraction** - Extracts color, pattern, sleeve length
3. **Scene Context Detection (PLACES365)** - Identifies the environment/scene
4. **LLM Intent Reasoning** - Generates filtering criteria based on attributes + scene
5. **Product Retrieval** - Fetches matching products from the database

### 4. **Enhanced UI Components**

#### Input Mode Selector
- **Video Mode Button** - Switches to video capture interface
- **Photo Mode Button** - Switches to photo upload interface
- Visual feedback (blue = active, grey = inactive)

#### Conditional Rendering
- Video player shown only in Video Mode
- Photo upload form shown only in Photo Mode
- Displays either captured frame or uploaded photo based on active mode
- Action buttons adapt based on current input mode

#### New Controls
- **"Pause & Capture Frame"** - Pauses video and captures current frame (Video Mode only)
- **"Detect Fashion Items"** - Processes the image through YOLOv8 (works for both modes)
- **"Identify Scene"** - Processes the image through Places365 (works for both modes)

### 5. **State Management**
Added new state variables:
- `inputMode` - Tracks whether user is in "video" or "photo" mode
- `uploadedPhoto` - Stores the uploaded photo as base64 string
- `fileInputRef` - Reference to the file input element

When switching modes, all downstream processing states are automatically reset:
- Detections
- Selected item
- Attributes
- Scene
- LLM output
- Products

### 6. **Backend Integration**
No backend changes required. The existing endpoints handle both sources:
- `/detect` - Works with base64 images from either video or photos
- `/extract-attributes` - Works with cropped images from both sources
- `/scene` - Works with base64 images from either source
- `/llm` - Uses extracted attributes (works identically)
- `/search` - Uses LLM filters (works identically)

## File Modified
- `frontend/src/components/VideoPlayer.jsx` - Enhanced with photo upload and auto-play

## Testing Instructions

1. **Test Auto-Play:**
   - Open the app in browser
   - Video should auto-play after component mounts

2. **Test Video Mode:**
   - Click "Video Mode" button (should already be selected)
   - Pause the video at any frame
   - Click "Pause & Capture Frame"
   - Click "Detect Fashion Items"
   - Verify full processing pipeline works

3. **Test Photo Mode:**
   - Click "Photo Mode" button
   - Upload an image of clothing/fashion item
   - Click "Detect Fashion Items"
   - Verify full processing pipeline works identically to video

4. **Test Mode Switching:**
   - Switch between modes
   - Verify all states reset properly
   - Verify UI updates correctly

## Notes
- Photo upload resets all downstream states to ensure clean analysis
- Both video and photo use the same base64 encoding format for backend compatibility
- Error handling includes user-friendly alerts
- Console logging helps with debugging the flow
