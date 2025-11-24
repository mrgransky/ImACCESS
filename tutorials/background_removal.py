from withoutbg import WithoutBG

# Local processing with Open Source model
model = WithoutBG.opensource()
result = model.remove_background("document_icon.png")
result.save("document_icon_output.png")

# # withoutBG Pro for best quality
# model = WithoutBG.api(api_key="sk_your_key")
# result = model.remove_background("input.jpg")
# result.save("document_icon_output.png")